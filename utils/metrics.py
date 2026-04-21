# utils/metrics.py

import torch
import numpy as np


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean((pred - target) ** 2).item()


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = compute_mse(pred, target)
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(torch.tensor(1.0 / mse)).item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute mean SSIM over a batch of images.
    pred, target: (B, 3, H, W) float tensors in [0, 1]
    """
    from skimage.metrics import structural_similarity as sk_ssim

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    scores = []
    for p, t in zip(pred_np, target_np):
        # skimage expects (H, W, C) and channel_axis kwarg
        p_hwc = np.transpose(p, (1, 2, 0))
        t_hwc = np.transpose(t, (1, 2, 0))
        score = sk_ssim(p_hwc, t_hwc, data_range=1.0, channel_axis=-1)
        scores.append(score)

    return float(np.mean(scores))


def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Return MSE, PSNR, and SSIM for a batch."""
    return {
        "mse": compute_mse(pred, target),
        "psnr": compute_psnr(pred, target),
        "ssim": compute_ssim(pred, target),
    }
