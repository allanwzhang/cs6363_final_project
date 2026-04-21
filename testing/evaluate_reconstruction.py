# testing/evaluate_reconstruction.py

import argparse
import torch
from tqdm import tqdm

from data.celeba_dataset import build_celeba_dataloader
from data.transforms import get_basic_image_transform
from model.clip_wrapper import CLIPVisionWrapper
from model.decoders import TokenGridDecoder
from utils.metrics import compute_mse


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = get_basic_image_transform(224)

    _, loader = build_celeba_dataloader(
        root=args.dataset_root,
        split="valid",
        transform=transform,
        batch_size=args.batch_size,
        shuffle=False,
    )

    clip_model = CLIPVisionWrapper(device=device)

    decoder = TokenGridDecoder().to(device)
    decoder.load_state_dict(torch.load(args.decoder_path))
    decoder.eval()

    total_mse = 0

    with torch.no_grad():

        for batch in tqdm(loader):

            images = batch["image"].to(device)

            features = clip_model.extract_features(
                images=images,
                layers=[args.layer],
            )

            recon = decoder(features.hidden_states[args.layer])

            total_mse += compute_mse(recon, images)

    print("Average MSE:", total_mse / len(loader))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", default="./datasets")
    parser.add_argument("--decoder_path", required=True)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    main(args)