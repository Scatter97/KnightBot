from __future__ import annotations

import argparse

import torch

from halfkp_512 import (
    HalfKP512,
    quantize_model,
    save_quantized_network,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
    )

    model = HalfKP512()

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    quantized = quantize_model(
        model
    )

    save_quantized_network(
        quantized,
        args.output,
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Exported: {args.output}"
    )


if __name__ == "__main__":
    main()