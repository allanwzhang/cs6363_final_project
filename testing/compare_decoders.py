# testing/compare_decoders.py
#
# Trains all three decoder architectures on CLIP hidden states from layers
# 1, 4, 8, 12 using a small CelebA subset, evaluates MSE on a validation
# subset, then saves:
#   - outputs/decoder_comparison/results.txt  (MSE table)
#   - outputs/decoder_comparison/viz_compare.png  (grid: rows=decoders, cols=layers)

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.celeba_dataset import build_celeba_dataloader
from data.transforms import get_basic_image_transform
from model.clip_wrapper import CLIPVisionWrapper
from model.decoders import TokenGridDecoder, MLPImageDecoder, PooledEmbeddingDecoder


OUTPUT_SIZE = 64
TOKEN_DIM = 768
SEQ_LEN = 50  # 1 CLS + 49 patches


def make_decoder(arch: str) -> nn.Module:
    if arch == "TokenGrid":
        return TokenGridDecoder(output_size=OUTPUT_SIZE)
    elif arch == "MLP":
        return MLPImageDecoder(seq_len=SEQ_LEN, token_dim=TOKEN_DIM, output_size=OUTPUT_SIZE)
    elif arch == "Pooled":
        return PooledEmbeddingDecoder(embedding_dim=TOKEN_DIM, output_size=OUTPUT_SIZE)
    else:
        raise ValueError(f"Unknown arch: {arch}")


def get_decoder_input(arch: str, hidden_state: torch.Tensor) -> torch.Tensor:
    """
    Each decoder expects a different input format:
      TokenGrid  -> (B, 50, 768)  full token sequence (CLS + patches)
      MLP        -> (B, 50, 768)  full token sequence (flattened internally)
      Pooled     -> (B, 768)      CLS token only (position 0), as the per-layer "pooled" rep
    """
    if arch in ("TokenGrid", "MLP"):
        return hidden_state
    elif arch == "Pooled":
        return hidden_state[:, 0, :]  # CLS token


def resize_target(images: torch.Tensor) -> torch.Tensor:
    return F.interpolate(images, size=(OUTPUT_SIZE, OUTPUT_SIZE), mode="bilinear", align_corners=False)


def train(decoder, arch, clip_model, loader, optimizer, device, layer):
    decoder.train()
    total_loss = 0.0
    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device)
        with torch.no_grad():
            feats = clip_model.extract_features(images=images, layers=[layer])
        x = get_decoder_input(arch, feats.hidden_states[layer])
        recon = decoder(x)
        target = resize_target(images)
        loss = F.mse_loss(recon, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(decoder, arch, clip_model, loader, device, layer):
    decoder.eval()
    total_mse = 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            feats = clip_model.extract_features(images=images, layers=[layer])
            x = get_decoder_input(arch, feats.hidden_states[layer])
            recon = decoder(x)
            target = resize_target(images)
            total_mse += F.mse_loss(recon, target).item()
    return total_mse / len(loader)


def reconstruct_batch(decoder, arch, clip_model, images, device, layer):
    decoder.eval()
    with torch.no_grad():
        feats = clip_model.extract_features(images=images.to(device), layers=[layer])
        x = get_decoder_input(arch, feats.hidden_states[layer])
        return decoder(x).cpu()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = get_basic_image_transform(224)

    _, train_loader_full = build_celeba_dataloader(
        root=args.dataset_root, split="train", transform=transform,
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    _, val_loader_full = build_celeba_dataloader(
        root=args.dataset_root, split="valid", transform=transform,
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    train_subset = Subset(train_loader_full.dataset, range(min(args.train_samples, len(train_loader_full.dataset))))
    val_subset = Subset(val_loader_full.dataset, range(min(args.val_samples, len(val_loader_full.dataset))))

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # grab a fixed small batch of images for the visual grid
    vis_batch = next(iter(DataLoader(val_subset, batch_size=4, shuffle=False, num_workers=0)))
    vis_images = vis_batch["image"]
    vis_target = resize_target(vis_images)

    clip_model = CLIPVisionWrapper(device=device, freeze=True)

    layers = args.layers
    archs = ["TokenGrid", "MLP", "Pooled"]

    # results[arch][layer] = eval_mse
    results = {arch: {} for arch in archs}
    # recons[arch][layer] = (4, 3, 64, 64) tensor
    recons = {arch: {} for arch in archs}

    for arch in archs:
        for layer in layers:
            print(f"\n--- Training {arch} on layer {layer} ---")
            decoder = make_decoder(arch).to(device)
            optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

            for epoch in range(args.epochs):
                train_loss = train(decoder, arch, clip_model, train_loader, optimizer, device, layer)
                print(f"  epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.5f}")

            mse = evaluate(decoder, arch, clip_model, val_loader, device, layer)
            results[arch][layer] = mse
            print(f"  eval MSE={mse:.5f}")

            recons[arch][layer] = reconstruct_batch(decoder, arch, clip_model, vis_images, device, layer)

    # Save results table
    results_path = out_dir / "results.txt"
    with open(results_path, "w") as f:
        header = f"{'Decoder':<14}" + "".join(f"  Layer {l:<6}" for l in layers)
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for arch in archs:
            row = f"{arch:<14}" + "".join(f"  {results[arch][l]:.5f}  " for l in layers)
            f.write(row + "\n")

    print(f"\n{'Decoder':<14}" + "".join(f"  Layer {l:<6}" for l in layers))
    print("-" * (14 + 16 * len(layers)))
    for arch in archs:
        print(f"{arch:<14}" + "".join(f"  {results[arch][l]:.5f}  " for l in layers))
    print(f"\nResults saved to {results_path}")

    # Build visualization grid
    # rows: Original + each decoder  (1 + 3 = 4 rows)
    # cols: one per layer
    n_samples = 4
    n_cols = len(layers)
    row_labels = ["Original"] + archs
    n_rows = len(row_labels)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3 * n_cols, 3 * n_rows),
        squeeze=False,
    )

    for col, layer in enumerate(layers):
        axes[0][col].set_title(f"Layer {layer}", fontsize=11, fontweight="bold")
        img_grid = _make_row_strip(vis_target, n_samples)
        axes[0][col].imshow(img_grid)
        axes[0][col].axis("off")

        for row, arch in enumerate(archs, start=1):
            recon_grid = _make_row_strip(recons[arch][layer], n_samples)
            mse_val = results[arch][layer]
            axes[row][col].imshow(recon_grid)
            axes[row][col].set_xlabel(f"MSE={mse_val:.4f}", fontsize=8)
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])

    for row, label in enumerate(row_labels):
        axes[row][0].set_ylabel(label, fontsize=10, fontweight="bold", rotation=90, labelpad=8)

    plt.suptitle("Decoder architecture comparison across CLIP layers", fontsize=13, y=1.01)
    plt.tight_layout()

    viz_path = out_dir / "viz_compare.png"
    plt.savefig(viz_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved to {viz_path}")


def _make_row_strip(images: torch.Tensor, n: int) -> "np.ndarray":
    """Concatenate n images side by side into one wide image."""
    import numpy as np
    imgs = [images[i].permute(1, 2, 0).clamp(0, 1).numpy() for i in range(n)]
    return np.concatenate(imgs, axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="./datasets")
    parser.add_argument("--output_dir", default="./outputs/decoder_comparison")
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 4, 8, 12])
    parser.add_argument("--train_samples", type=int, default=512)
    parser.add_argument("--val_samples", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
