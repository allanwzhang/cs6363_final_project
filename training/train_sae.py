# training/train_sae.py

import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from data.celeba_dataset import build_celeba_dataloader
from data.transforms import get_basic_image_transform
from model.clip_wrapper import CLIPVisionWrapper
from model.sparse_autoencoder import SparseAutoencoder


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = get_basic_image_transform(224)

    _, loader = build_celeba_dataloader(
        root=args.dataset_root,
        split="train",
        transform=transform,
        batch_size=args.batch_size,
    )

    clip_model = CLIPVisionWrapper(device=device)

    sae = SparseAutoencoder(
        input_dim=50 * 768,
        latent_dim=args.latent_dim,
    ).to(device)

    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

    for epoch in range(args.epochs):

        total_loss = 0

        for batch in tqdm(loader):

            images = batch["image"].to(device)

            features = clip_model.extract_features(
                images=images,
                layers=[args.layer],
            )

            tokens = features.hidden_states[args.layer]
            tokens = tokens.flatten(start_dim=1)

            out = sae(tokens)

            optimizer.zero_grad()
            out.total_loss.backward()
            optimizer.step()

            total_loss += out.total_loss.item()

        print("Epoch:", epoch, "Loss:", total_loss / len(loader))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", default="./datasets")
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    main(args)