# testing/clip_smoketest.py

import torch

from data.celeba_dataset import build_celeba_dataloader
from data.transforms import get_basic_image_transform
from model.clip_wrapper import CLIPVisionWrapper


def main():

    transform = get_basic_image_transform(224)

    _, loader = build_celeba_dataloader(
        root="./datasets",
        split="train",
        transform=transform,
        batch_size=4,
    )

    batch = next(iter(loader))

    model = CLIPVisionWrapper(device="cpu")

    features = model.extract_features(
        images=batch["image"],
        layers=[4, 12],
    )

    print("Input:", batch["image"].shape)
    print("Layer 4:", features.hidden_states[4].shape)
    print("Layer 12:", features.hidden_states[12].shape)
    print("Pooler:", features.pooler_output.shape)
    print("Grid size:", model.grid_size)


if __name__ == "__main__":
    main()