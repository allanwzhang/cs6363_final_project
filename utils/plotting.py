# utils/plotting.py

import matplotlib.pyplot as plt
import torch


def tensor_to_img(x):

    x = x.detach().cpu()

    if x.dim() == 3:
        x = x.permute(1, 2, 0)

    return x.clamp(0, 1)


def show_reconstruction_grid(images, reconstructions, n=6):

    plt.figure(figsize=(12, 4))

    for i in range(n):

        plt.subplot(2, n, i + 1)
        plt.imshow(tensor_to_img(images[i]))
        plt.axis("off")

        plt.subplot(2, n, n + i + 1)
        plt.imshow(tensor_to_img(reconstructions[i]))
        plt.axis("off")

    plt.tight_layout()
    plt.show()