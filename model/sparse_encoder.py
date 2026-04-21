# model/sparse_autoencoder.py

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEOutput:
    reconstruction: torch.Tensor
    codes: torch.Tensor
    mse_loss: torch.Tensor
    sparsity_loss: torch.Tensor
    total_loss: torch.Tensor


class SparseAutoencoder(nn.Module):
    """
    Simple sparse autoencoder for later analysis.

    Intended for:
      - pooled CLIP features, shape (B, D)
      - or flattened token features, shape (B, T*D)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        sparsity_weight: float = 1e-3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sparsity_weight = sparsity_weight

        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> SAEOutput:
        z = self.encode(x)
        x_hat = self.decode(z)

        mse_loss = F.mse_loss(x_hat, x)
        sparsity_loss = z.abs().mean()
        total_loss = mse_loss + self.sparsity_weight * sparsity_loss

        return SAEOutput(
            reconstruction=x_hat,
            codes=z,
            mse_loss=mse_loss,
            sparsity_loss=sparsity_loss,
            total_loss=total_loss,
        )

    @torch.no_grad()
    def get_top_activations(self, x: torch.Tensor, top_k: int = 10):
        z = self.encode(x)
        values, indices = torch.topk(z, k=top_k, dim=-1)
        return values, indices