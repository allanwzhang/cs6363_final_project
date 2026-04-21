# testing/evaluate_reconstruction.py

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from data.celeba_dataset import build_celeba_dataloader
from data.transforms import get_basic_image_transform
from model.clip_wrapper import CLIPVisionWrapper
from model.decoders import TokenGridDecoder
from utils.metrics import compute_all_metrics


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
    decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
    decoder.eval()

    total_mse = total_psnr = total_ssim = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader):
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

            m = compute_all_metrics(recon, target)
            total_mse += m["mse"]
            total_psnr += m["psnr"]
            total_ssim += m["ssim"]
            n_batches += 1

    results = {
        "layer": args.layer,
        "decoder": "TokenGridDecoder",
        "n_batches": n_batches,
        "mse": round(total_mse / n_batches, 6),
        "psnr_db": round(total_psnr / n_batches, 4),
        "ssim": round(total_ssim / n_batches, 4),
    }

    print(f"\nLayer {args.layer} results:")
    print(f"  MSE:  {results['mse']:.6f}")
    print(f"  PSNR: {results['psnr_db']:.2f} dB")
    print(f"  SSIM: {results['ssim']:.4f}")

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_path = out / f"eval_layer{args.layer}.json"
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {save_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", default="./datasets")
    parser.add_argument("--decoder_path", required=True)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", default="./outputs/eval")

    args = parser.parse_args()
    main(args)
