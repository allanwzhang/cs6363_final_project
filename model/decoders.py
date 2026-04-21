# model/decoders.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPImageDecoder(nn.Module):
    """
    Baseline decoder:
      token sequence -> flatten -> MLP -> image
    """

    def __init__(
        self,
        seq_len: int = 50,
        token_dim: int = 768,
        output_size: int = 64,
        hidden_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.output_dim = 3 * output_size * output_size

        self.net = nn.Sequential(
            nn.Linear(seq_len * token_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        x = x.reshape(bsz, -1)
        x = self.net(x)
        x = x.view(bsz, 3, self.output_size, self.output_size)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TokenGridDecoder(nn.Module):
    """
    Main decoder for CLIP token hidden states.

    Expected CLIP ViT-B/32 token input:
      (B, 50, 768)

    Workflow:
      1. remove CLS token -> (B, 49, 768)
      2. reshape to 7x7 grid
      3. upsample to output image
    """

    def __init__(
        self,
        token_dim: int = 768,
        grid_size: int = 7,
        output_size: int = 64,
        remove_cls_token: bool = True,
        base_channels: int = 256,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.grid_size = grid_size
        self.output_size = output_size
        self.remove_cls_token = remove_cls_token

        self.input_proj = nn.Sequential(
            nn.Conv2d(token_dim, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.stage1 = ConvBlock(base_channels, base_channels)
        self.stage2 = ConvBlock(base_channels, base_channels // 2)
        self.stage3 = ConvBlock(base_channels // 2, base_channels // 4)
        self.stage4 = ConvBlock(base_channels // 4, base_channels // 8)

        self.to_rgb = nn.Sequential(
            nn.Conv2d(base_channels // 8, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def _tokens_to_grid(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B,T,D), got {tuple(x.shape)}")

        if self.remove_cls_token:
            x = x[:, 1:, :]

        bsz, num_tokens, dim = x.shape
        expected = self.grid_size * self.grid_size
        if num_tokens != expected:
            raise ValueError(
                f"Expected {expected} patch tokens after CLS removal, got {num_tokens}"
            )

        return x.transpose(1, 2).reshape(bsz, dim, self.grid_size, self.grid_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._tokens_to_grid(x)
        x = self.input_proj(x)

        x = self.stage1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # 7 -> 14

        x = self.stage2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # 14 -> 28

        x = self.stage3(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # 28 -> 56

        x = self.stage4(x)
        x = F.interpolate(
            x, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False
        )  # 56 -> output_size

        x = self.to_rgb(x)
        return x


class PooledEmbeddingDecoder(nn.Module):
    """
    Decoder for pooled CLIP features:
      (B, D) -> image
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        output_size: int = 64,
        hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.output_size = output_size

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 8 * 8 * 128),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 8->16
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 16->32
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 32->64
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        x = self.fc(x)
        x = x.view(bsz, 128, 8, 8)
        x = self.decoder(x)
        return x