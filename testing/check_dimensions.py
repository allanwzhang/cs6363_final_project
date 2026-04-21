import argparse

import torch

from model.decoders import TokenGridDecoder


def main(args):
    bsz = args.batch_size
    seq_len = args.seq_len
    token_dim = args.token_dim
    output_size = args.output_size

    tokens = torch.randn(bsz, seq_len, token_dim)
    decoder = TokenGridDecoder(
        token_dim=token_dim,
        grid_size=args.grid_size,
        output_size=output_size,
        remove_cls_token=args.remove_cls_token,
    )
    recon = decoder(tokens)

    print("Input tokens shape:", tuple(tokens.shape))
    print("Reconstruction shape:", tuple(recon.shape))

    if recon.shape != (bsz, 3, output_size, output_size):
        raise RuntimeError("Decoder output shape does not match expected image shape.")

    print("Dimension check passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--token_dim", type=int, default=768)
    parser.add_argument("--grid_size", type=int, default=7)
    parser.add_argument("--output_size", type=int, default=64)
    parser.add_argument(
        "--remove-cls-token",
        dest="remove_cls_token",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    main(args)
