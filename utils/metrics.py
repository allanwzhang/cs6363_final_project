# utils/metrics.py

import torch


def compute_mse(pred, target):
    return torch.mean((pred - target) ** 2).item()


def compute_psnr(pred, target):
    mse = compute_mse(pred, target)
    return 10 * torch.log10(torch.tensor(1.0 / mse)).item()