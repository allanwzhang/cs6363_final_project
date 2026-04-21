# data/inspect_data.py

import math
import matplotlib.pyplot as plt
import torch

from celeba_dataset import build_celeba_dataset, print_dataset_info
from transforms import get_basic_image_transform


def tensor_to_image(tensor):
    """
    Convert a CHW tensor to HWC format for matplotlib.
    Assumes tensor values are in [0, 1].
    """
    image = tensor.detach().cpu()

    if image.dim() == 3 and image.shape[0] in [1, 3]:
        image = image.permute(1, 2, 0)

    return image.clamp(0, 1)


def show_samples(dataset, num_samples=6):
    """
    Display a few samples from the dataset.
    """
    cols = 3
    rows = math.ceil(num_samples / cols)

    plt.figure(figsize=(12, 4 * rows))

    for i in range(num_samples):
        sample = dataset[i]
        image = tensor_to_image(sample["image"])
        filename = sample["filename"]

        plt.subplot(rows, cols, i + 1)
        plt.imshow(image)
        plt.title(filename, fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    root = "./datasets"

    dataset = build_celeba_dataset(
        root=root,
        split="train",
        target_type="attr",
        transform=get_basic_image_transform(image_size=224),
        download=False,
    )

    print_dataset_info(dataset)
    show_samples(dataset, num_samples=6)


if __name__ == "__main__":
    main()