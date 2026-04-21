# model/clip_wrapper.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Union

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel, CLIPVisionModelWithProjection


TensorOrImage = Union[torch.Tensor, Image.Image]
BatchInput = Union[torch.Tensor, Sequence[TensorOrImage], TensorOrImage]


@dataclass
class CLIPVisionFeatures:
    pixel_values: torch.Tensor
    hidden_states: Dict[int, torch.Tensor]
    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor
    image_embeds: Optional[torch.Tensor] = None


class CLIPVisionWrapper(nn.Module):
    """
    Wrapper for CLIP vision encoder.

    Designed to work with your CelebA dataloader, where:
      - batch["image"] is a float tensor
      - shape is typically (B, 3, H, W)
      - values are in [0, 1]

    Notes on hidden_states indexing:
      hidden_states[0]  = embedding output
      hidden_states[1]  = after transformer block 1
      ...
      hidden_states[12] = after transformer block 12 for ViT-B/32
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[Union[str, torch.device]] = None,
        with_projection: bool = False,
        freeze: bool = True,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.with_projection = with_projection

        self.processor = CLIPImageProcessor.from_pretrained(model_name)

        if with_projection:
            self.model = CLIPVisionModelWithProjection.from_pretrained(model_name)
        else:
            self.model = CLIPVisionModel.from_pretrained(model_name)

        self.model.to(self.device)

        if freeze:
            self.freeze()

        self.hidden_size = self.model.config.hidden_size
        self.image_size = self.model.config.image_size
        self.patch_size = self.model.config.patch_size
        self.num_hidden_layers = self.model.config.num_hidden_layers
        self.grid_size = self.image_size // self.patch_size
        self.num_patch_tokens = self.grid_size * self.grid_size
        self.sequence_length = self.num_patch_tokens + 1  # + CLS token

    def freeze(self) -> None:
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = True
        self.model.train()

    def preprocess(self, images: BatchInput) -> torch.Tensor:
        """
        Converts input into CLIP pixel_values.

        Accepts:
          - single tensor (C,H,W)
          - batch tensor (B,C,H,W)
          - single PIL image
          - list of tensors/PIL images

        Returns:
          pixel_values: (B, 3, 224, 224)
        """
        if isinstance(images, torch.Tensor):
            if images.ndim == 3:
                images = [images.cpu()]
            elif images.ndim == 4:
                images = [img.cpu() for img in images]
            else:
                raise ValueError(
                    f"Expected image tensor with shape (C,H,W) or (B,C,H,W), got {tuple(images.shape)}"
                )
        elif isinstance(images, Image.Image):
            images = [images]
        else:
            images = list(images)

        processed = self.processor(images=images, return_tensors="pt")
        return processed["pixel_values"].to(self.device)

    @torch.no_grad()
    def extract_features(
        self,
        images: Optional[BatchInput] = None,
        pixel_values: Optional[torch.Tensor] = None,
        layers: Optional[Iterable[int]] = None,
    ) -> CLIPVisionFeatures:
        """
        Run CLIP vision encoder and return requested hidden states.

        Pass exactly one of:
          - images=...
          - pixel_values=...
        """
        if (images is None) == (pixel_values is None):
            raise ValueError("Pass exactly one of `images` or `pixel_values`.")

        if pixel_values is None:
            pixel_values = self.preprocess(images)
        else:
            if pixel_values.ndim == 3:
                pixel_values = pixel_values.unsqueeze(0)
            pixel_values = pixel_values.to(self.device)

        outputs = self.model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        all_hidden_states = outputs.hidden_states
        if all_hidden_states is None:
            raise RuntimeError("CLIP did not return hidden states.")

        if layers is None:
            layers = range(len(all_hidden_states))

        hidden_dict: Dict[int, torch.Tensor] = {}
        for idx in layers:
            if idx < 0 or idx >= len(all_hidden_states):
                raise ValueError(
                    f"Requested layer {idx}, valid range is 0..{len(all_hidden_states)-1}"
                )
            hidden_dict[idx] = all_hidden_states[idx]

        image_embeds = getattr(outputs, "image_embeds", None)

        return CLIPVisionFeatures(
            pixel_values=pixel_values,
            hidden_states=hidden_dict,
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=outputs.pooler_output,
            image_embeds=image_embeds,
        )

    @staticmethod
    def remove_cls_token(tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, 1+N, D)
        returns: (B, N, D)
        """
        if tokens.ndim != 3:
            raise ValueError(f"Expected (B,T,D), got {tuple(tokens.shape)}")
        return tokens[:, 1:, :]

    def tokens_to_grid(self, tokens: torch.Tensor, remove_cls: bool = True) -> torch.Tensor:
        """
        Convert token sequence into spatial grid.

        For clip-vit-base-patch32:
          (B, 50, 768) -> remove CLS -> (B, 49, 768) -> (B, 768, 7, 7)
        """
        if tokens.ndim != 3:
            raise ValueError(f"Expected (B,T,D), got {tuple(tokens.shape)}")

        if remove_cls and tokens.shape[1] == self.sequence_length:
            tokens = self.remove_cls_token(tokens)

        bsz, num_tokens, dim = tokens.shape
        expected = self.num_patch_tokens
        if num_tokens != expected:
            raise ValueError(
                f"Expected {expected} patch tokens, got {num_tokens}. "
                "Check whether CLS removal matches your input."
            )

        return tokens.transpose(1, 2).reshape(bsz, dim, self.grid_size, self.grid_size)