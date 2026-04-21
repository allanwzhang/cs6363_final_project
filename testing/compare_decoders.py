# testing/compare_decoders.py
#
# Trains all three decoder architectures on CLIP hidden states from layers
# 1, 4, 8, 12 using a CelebA subset, evaluates MSE / PSNR / SSIM, then saves:
#   - outputs/full_run/{arch}_layer{N}.pt         (checkpoint per decoder per layer)
#   - outputs/full_run/results.csv                (full metrics table)
#   - outputs/full_run/results.txt                (human-readable table)
#   - outputs/full_run/viz_compare.png            (image grid: rows=decoders, cols=layers)
#   - outputs/full_run/bar_mse.png                (bar chart — MSE per layer per decoder)
#   - outputs/full_run/bar_psnr.png               (bar chart — PSNR per layer per decoder)
#   - outputs/full_run/bar_ssim.png               (bar chart — SSIM per layer per decoder)

import argparse
import csv
import os
from pathlib import Path

import numpy as np
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
from utils.metrics import compute_all_metrics


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
      TokenGrid  -> (B, 50, 768)  full token sequence
      MLP        -> (B, 50, 768)  full token sequence (flattened internally)
      Pooled     -> (B, 768)      CLS token only (position 0)
    """
    if arch in ("TokenGrid", "MLP"):
        return hidden_state
    elif arch == "Pooled":
        return hidden_state[:, 0, :]


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
    totals = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0}
    n = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            feats = clip_model.extract_features(images=images, layers=[layer])
            x = get_decoder_input(arch, feats.hidden_states[layer])
            recon = decoder(x)
            target = resize_target(images)
            m = compute_all_metrics(recon.cpu(), target.cpu())
            for k in totals:
                totals[k] += m[k]
            n += 1
    return {k: v / n for k, v in totals.items()}


def reconstruct_batch(decoder, arch, clip_model, images, device, layer):
    decoder.eval()
    with torch.no_grad():
        feats = clip_model.extract_features(images=images.to(device), layers=[layer])
        x = get_decoder_input(arch, feats.hidden_states[layer])
        return decoder(x).cpu()


def save_bar_chart(results, layers, archs, metric, ylabel, title, save_path):
    x = np.arange(len(layers))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))

    for i, arch in enumerate(archs):
        vals = [results[arch][layer][metric] for layer in layers]
        bars = ax.bar(x + i * width, vals, width, label=arch)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xlabel("CLIP Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

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

    vis_batch = next(iter(DataLoader(val_subset, batch_size=4, shuffle=False, num_workers=0)))
    vis_images = vis_batch["image"]
    vis_target = resize_target(vis_images)

    clip_model = CLIPVisionWrapper(device=device, freeze=True)

    layers = args.layers
    archs = ["TokenGrid", "MLP", "Pooled"]

    # results[arch][layer] = {mse, psnr, ssim}
    results = {arch: {} for arch in archs}
    recons = {arch: {} for arch in archs}

    for arch in archs:
        for layer in layers:
            print(f"\n--- Training {arch} on layer {layer} ---")
            decoder = make_decoder(arch).to(device)
            optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

            for epoch in range(args.epochs):
                train_loss = train(decoder, arch, clip_model, train_loader, optimizer, device, layer)
                print(f"  epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.5f}")

            metrics = evaluate(decoder, arch, clip_model, val_loader, device, layer)
            results[arch][layer] = metrics
            print(f"  MSE={metrics['mse']:.5f}  PSNR={metrics['psnr']:.2f}dB  SSIM={metrics['ssim']:.4f}")

            ckpt_path = out_dir / f"{arch}_layer{layer}.pt"
            torch.save(decoder.state_dict(), ckpt_path)
            print(f"  Checkpoint: {ckpt_path}")

            recons[arch][layer] = reconstruct_batch(decoder, arch, clip_model, vis_images, device, layer)

    # ── Save CSV ────────────────────────────────────────────────────────────────
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["decoder", "layer", "mse", "psnr_db", "ssim"])
        for arch in archs:
            for layer in layers:
                m = results[arch][layer]
                writer.writerow([arch, layer, round(m["mse"], 6), round(m["psnr"], 4), round(m["ssim"], 4)])
    print(f"\nCSV saved: {csv_path}")

    # ── Save human-readable text table ──────────────────────────────────────────
    txt_path = out_dir / "results.txt"
    with open(txt_path, "w") as f:
        header = f"\n{'Decoder':<14}" + "".join(
            f"  {'L'+str(l)+' MSE':<10}{'PSNR':>7}{'SSIM':>7}  " for l in layers
        )
        f.write(header + "\n" + "-" * len(header) + "\n")
        for arch in archs:
            row = f"{arch:<14}"
            for layer in layers:
                m = results[arch][layer]
                row += f"  {m['mse']:.5f}   {m['psnr']:>5.2f}  {m['ssim']:.4f}  "
            f.write(row + "\n")

    with open(txt_path) as f:
        print(f.read())
    print(f"Text table saved: {txt_path}")

    # ── Bar charts ───────────────────────────────────────────────────────────────
    save_bar_chart(
        results, layers, archs, "mse",
        "MSE (lower is better)",
        "Reconstruction MSE by decoder and CLIP layer",
        out_dir / "bar_mse.png",
    )
    save_bar_chart(
        results, layers, archs, "psnr",
        "PSNR dB (higher is better)",
        "Reconstruction PSNR by decoder and CLIP layer",
        out_dir / "bar_psnr.png",
    )
    save_bar_chart(
        results, layers, archs, "ssim",
        "SSIM (higher is better)",
        "Reconstruction SSIM by decoder and CLIP layer",
        out_dir / "bar_ssim.png",
    )

    # ── Visual reconstruction grid ───────────────────────────────────────────────
    n_samples = 4
    row_labels = ["Original"] + archs
    fig, axes = plt.subplots(
        len(row_labels), len(layers),
        figsize=(3 * len(layers), 3 * len(row_labels)),
        squeeze=False,
    )

    for col, layer in enumerate(layers):
        axes[0][col].set_title(f"Layer {layer}", fontsize=11, fontweight="bold")
        axes[0][col].imshow(_make_row_strip(vis_target, n_samples))
        axes[0][col].axis("off")

        for row, arch in enumerate(archs, start=1):
            m = results[arch][layer]
            recon_grid = _make_row_strip(recons[arch][layer], n_samples)
            axes[row][col].imshow(recon_grid)
            axes[row][col].set_xlabel(
                f"MSE={m['mse']:.4f}  PSNR={m['psnr']:.1f}dB  SSIM={m['ssim']:.3f}",
                fontsize=7,
            )
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])

    for row, label in enumerate(row_labels):
        axes[row][0].set_ylabel(label, fontsize=10, fontweight="bold", rotation=90, labelpad=8)

    plt.suptitle("Decoder architecture comparison across CLIP layers", fontsize=13, y=1.01)
    plt.tight_layout()
    viz_path = out_dir / "viz_compare.png"
    plt.savefig(viz_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved: {viz_path}")


def _make_row_strip(images: torch.Tensor, n: int) -> np.ndarray:
    imgs = [images[i].permute(1, 2, 0).clamp(0, 1).numpy() for i in range(n)]
    return np.concatenate(imgs, axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="./datasets")
    parser.add_argument("--output_dir", default="./outputs/full_run")
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 4, 8, 12])
    parser.add_argument("--train_samples", type=int, default=512)
    parser.add_argument("--val_samples", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
