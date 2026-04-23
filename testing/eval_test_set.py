# testing/eval_test_set.py
#
# Loads saved decoder checkpoints from colab_results/ and evaluates them on
# the CelebA *test* split (the partition after train=162,770 and val=19,867).
# No training is performed.
#
# Outputs to test_results/ (or --output_dir):
#   test_results/results.csv
#   test_results/results.txt
#   test_results/viz_compare.png
#   test_results/bar_mse.png
#   test_results/bar_psnr.png
#   test_results/bar_ssim.png

import argparse
import csv
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so that data/, model/, utils/ are importable
# regardless of how or from where this script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
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
    TokenGrid / MLP -> (B, 50, 768) full token sequence
    Pooled          -> (B, 768)     CLS token only (position 0)
    """
    if arch in ("TokenGrid", "MLP"):
        return hidden_state
    elif arch == "Pooled":
        return hidden_state[:, 0, :]


def resize_target(images: torch.Tensor) -> torch.Tensor:
    return F.interpolate(images, size=(OUTPUT_SIZE, OUTPUT_SIZE), mode="bilinear", align_corners=False)


def evaluate(decoder, arch, clip_model, loader, device, layer):
    decoder.eval()
    totals = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0}
    n = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{arch} layer {layer}", leave=False):
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


def _make_row_strip(images: torch.Tensor, n: int) -> np.ndarray:
    imgs = [images[i].permute(1, 2, 0).clamp(0, 1).numpy() for i in range(n)]
    return np.concatenate(imgs, axis=1)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    transform = get_basic_image_transform(224)

    print("Loading CelebA test split...")
    _, test_loader = build_celeba_dataloader(
        root=args.dataset_root,
        split="test",
        transform=transform,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    print(f"Test set size: {len(test_loader.dataset)} images\n")

    vis_batch = next(iter(DataLoader(test_loader.dataset, batch_size=4, shuffle=False, num_workers=0)))
    vis_images = vis_batch["image"]
    vis_target = resize_target(vis_images)

    clip_model = CLIPVisionWrapper(device=device, freeze=True)

    layers = args.layers
    archs = ["TokenGrid", "MLP", "Pooled"]

    results = {arch: {} for arch in archs}
    recons = {arch: {} for arch in archs}

    for arch in archs:
        for layer in layers:
            ckpt_path = ckpt_dir / f"{arch}_layer{layer}.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: {ckpt_path}\n"
                    f"Make sure --checkpoint_dir points to the folder containing "
                    f"{arch}_layer{layer}.pt files."
                )

            print(f"--- Evaluating {arch} on layer {layer}  (checkpoint: {ckpt_path}) ---")
            decoder = make_decoder(arch).to(device)
            state_dict = torch.load(ckpt_path, map_location=device)
            decoder.load_state_dict(state_dict)

            metrics = evaluate(decoder, arch, clip_model, test_loader, device, layer)
            results[arch][layer] = metrics
            print(f"  MSE={metrics['mse']:.5f}  PSNR={metrics['psnr']:.2f}dB  SSIM={metrics['ssim']:.4f}\n")

            recons[arch][layer] = reconstruct_batch(decoder, arch, clip_model, vis_images, device, layer)

    # ── Save CSV ─────────────────────────────────────────────────────────────────
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["decoder", "layer", "mse", "psnr_db", "ssim"])
        for arch in archs:
            for layer in layers:
                m = results[arch][layer]
                writer.writerow([arch, layer, round(m["mse"], 6), round(m["psnr"], 4), round(m["ssim"], 4)])
    print(f"CSV saved: {csv_path}")

    # ── Save human-readable text table ───────────────────────────────────────────
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

    # ── Bar charts ────────────────────────────────────────────────────────────────
    save_bar_chart(
        results, layers, archs, "mse",
        "MSE (lower is better)",
        "Reconstruction MSE by decoder and CLIP layer (test set)",
        out_dir / "bar_mse.png",
    )
    save_bar_chart(
        results, layers, archs, "psnr",
        "PSNR dB (higher is better)",
        "Reconstruction PSNR by decoder and CLIP layer (test set)",
        out_dir / "bar_psnr.png",
    )
    save_bar_chart(
        results, layers, archs, "ssim",
        "SSIM (higher is better)",
        "Reconstruction SSIM by decoder and CLIP layer (test set)",
        out_dir / "bar_ssim.png",
    )

    # ── Visual reconstruction grid ────────────────────────────────────────────────
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

    plt.suptitle("Decoder architecture comparison across CLIP layers (test set)", fontsize=13, y=1.01)
    plt.tight_layout()
    viz_path = out_dir / "viz_compare.png"
    plt.savefig(viz_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved: {viz_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate saved decoder checkpoints on the CelebA test split. No training."
    )
    parser.add_argument(
        "--checkpoint_dir", default="./colab_results",
        help="Directory containing {arch}_layer{N}.pt checkpoints (default: ./colab_results)",
    )
    parser.add_argument(
        "--dataset_root", default="./datasets",
        help="Root directory of the CelebA dataset (default: ./datasets)",
    )
    parser.add_argument(
        "--output_dir", default="./test_results",
        help="Where to write evaluation outputs (default: ./test_results)",
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[1, 4, 8, 12],
        help="CLIP layers to evaluate (default: 1 4 8 12)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128,
        help="Batch size for evaluation (default: 128)",
    )
    args = parser.parse_args()
    main(args)
