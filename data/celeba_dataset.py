# data/celeba_dataset.py

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CelebA


class CelebAWrapper(CelebA):
    """
    Wrapper around torchvision.datasets.CelebA so each item returns:
      - image
      - attributes
      - index
      - filename

    This will make it easier later when saving reconstructions or
    analyzing specific samples.
    """

    def __getitem__(self, index):
        image, target = super().__getitem__(index)

        sample = {
            "image": image,
            "target": target,
            "index": index,
            "filename": self.filename[index],
        }
        return sample


def build_celeba_dataset(
    root: str,
    split: str = "train",
    target_type: str = "attr",
    transform=None,
    download: bool = False,
):
    """
    Builds a CelebA dataset.

    Args:
        root: path to dataset root
        split: 'train', 'valid', 'test', or 'all'
        target_type: usually 'attr' for CelebA attributes
        transform: torchvision transform
        download: whether to download automatically

    Returns:
        dataset
    """
    dataset = CelebAWrapper(
        root=root,
        split=split,
        target_type=target_type,
        transform=transform,
        download=download,
    )
    return dataset


def build_celeba_dataloader(
    root: str,
    split: str = "train",
    target_type: str = "attr",
    transform=None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    download: bool = False,
):
    """
    Builds dataset + dataloader for CelebA.
    """
    dataset = build_celeba_dataset(
        root=root,
        split=split,
        target_type=target_type,
        transform=transform,
        download=download,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return dataset, loader


def print_dataset_info(dataset):
    """
    Prints some useful info about the dataset.
    """
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print("Keys in sample:", sample.keys())
    print("Filename:", sample["filename"])
    print("Index:", sample["index"])

    if torch.is_tensor(sample["image"]):
        print("Image shape:", sample["image"].shape)
        print("Image dtype:", sample["image"].dtype)

    if torch.is_tensor(sample["target"]):
        print("Target shape:", sample["target"].shape)
        print("First 10 target values:", sample["target"][:10])


if __name__ == "__main__":
    from data.transforms import get_basic_image_transform

    dataset = build_celeba_dataset(
        root="./datasets",
        split="train",
        target_type="attr",
        transform=get_basic_image_transform(),
        download=False,
    )

    print_dataset_info(dataset)