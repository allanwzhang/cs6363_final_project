# testing/visualize_recons.py

import argparse
import torch
from torch.utils.data import DataLoader, Subset

from data.celeba_dataset import build_celeba_dataloader
from data.transforms import get_basic_image_transform
from model.clip_wrapper import CLIPVisionWrapper
from model.decoders import TokenGridDecoder
from utils.plotting import show_reconstruction_grid


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = get_basic_image_transform(224)

    _, loader = build_celeba_dataloader(
        root=args.dataset_root,
        split="valid",
        transform=transform,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    if args.max_samples > 0:
        subset_size = min(args.max_samples, len(loader.dataset))
        loader = DataLoader(
            Subset(loader.dataset, range(subset_size)),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
        )
        print(f"Using subset: {subset_size} samples")

    clip_model = CLIPVisionWrapper(device=device)

    decoder = TokenGridDecoder(output_size=args.output_size).to(device)
    decoder.load_state_dict(torch.load(args.decoder_path))
    decoder.eval()

    batch = next(iter(loader))

    images = batch["image"].to(device)

    features = clip_model.extract_features(
        images=images,
        layers=[args.layer],
    )

    recon = decoder(features.hidden_states[args.layer])

    target = torch.nn.functional.interpolate(
        images,
        size=(args.output_size, args.output_size),
        mode="bilinear",
        align_corners=False,
    )
    show_reconstruction_grid(target, recon)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", default="./datasets")
    parser.add_argument("--decoder_path", required=True)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()
    main(args)