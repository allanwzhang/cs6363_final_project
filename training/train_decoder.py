# training/train_decoder.py

import argparse
import csv
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data.celeba_dataset import build_celeba_dataloader
from data.transforms import get_basic_image_transform
from model.clip_wrapper import CLIPVisionWrapper
from model.decoders import TokenGridDecoder


def train_one_epoch(decoder, clip_model, loader, optimizer, device, layer):
    decoder.train()
    total_loss = 0

    for batch in tqdm(loader):
        images = batch["image"].to(device)

        features = clip_model.extract_features(images=images, layers=[layer])
        tokens = features.hidden_states[layer]

        recon = decoder(tokens)
        target = nn.functional.interpolate(
            images,
            size=(decoder.output_size, decoder.output_size),
            mode="bilinear",
            align_corners=False,
        )
        loss = nn.functional.mse_loss(recon, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = get_basic_image_transform(image_size=224)

    _, loader = build_celeba_dataloader(
        root=args.dataset_root,
        split="train",
        transform=transform,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    if args.max_samples > 0:
        subset_size = min(args.max_samples, len(loader.dataset))
        loader = DataLoader(
            Subset(loader.dataset, range(subset_size)),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
        )
        print(f"Using subset: {subset_size} samples")

    clip_model = CLIPVisionWrapper(device=device, freeze=True)

    decoder = TokenGridDecoder(output_size=args.output_size).to(device)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    save_path = Path(args.output_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    loss_log = []

    for epoch in range(args.epochs):
        loss = train_one_epoch(
            decoder, clip_model, loader, optimizer, device, args.layer,
        )
        print(f"Epoch {epoch+1}/{args.epochs}: loss={loss:.5f}")
        loss_log.append({"epoch": epoch + 1, "train_loss": loss})

        torch.save(
            decoder.state_dict(),
            save_path / f"decoder_layer{args.layer}.pt",
        )

    # Save loss curve as CSV
    csv_path = save_path / f"loss_layer{args.layer}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss"])
        writer.writeheader()
        writer.writerows(loss_log)
    print(f"Loss curve saved: {csv_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", default="./datasets")
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--output_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()
    main(args)
