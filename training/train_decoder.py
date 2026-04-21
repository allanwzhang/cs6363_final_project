# training/train_decoder.py

import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.celeba_dataset import build_celeba_dataloader
from data.transforms import get_basic_image_transform
from model.clip_wrapper import CLIPVisionWrapper
from model.decoders import TokenGridDecoder
from utils.metrics import compute_mse


def train_one_epoch(decoder, clip_model, loader, optimizer, device, layer):
    decoder.train()
    total_loss = 0

    for batch in tqdm(loader):
        images = batch["image"].to(device)

        features = clip_model.extract_features(images=images, layers=[layer])
        tokens = features.hidden_states[layer]

        recon = decoder(tokens)
        loss = nn.functional.mse_loss(recon, images)

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
    )

    clip_model = CLIPVisionWrapper(device=device, freeze=True)

    decoder = TokenGridDecoder().to(device)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    for epoch in range(args.epochs):

        loss = train_one_epoch(
            decoder,
            clip_model,
            loader,
            optimizer,
            device,
            args.layer,
        )

        print(f"Epoch {epoch+1}: loss={loss:.5f}")

        save_path = Path(args.output_dir)
        save_path.mkdir(exist_ok=True, parents=True)

        torch.save(
            decoder.state_dict(),
            save_path / f"decoder_layer{args.layer}.pt",
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", default="./datasets")
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", default="./outputs")

    args = parser.parse_args()
    main(args)