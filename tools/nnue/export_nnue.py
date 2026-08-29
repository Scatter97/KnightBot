from __future__ import annotations

import argparse
from pathlib import Path

import torch

from knightbot_nnue import (
    KnightBotNNUE,
    quantize_model,
    save_quantized_network,
)


def parse_args() -> argparse.Namespace:
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


def main() -> None:
    args = parse_args()

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
    )

    model = KnightBotNNUE()

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    network = quantize_model(
        model
    )

    output_path = Path(
        args.output
    )

    save_quantized_network(
        network,
        output_path,
    )

    print(
        f"Exported {output_path}"
    )


if __name__ == "__main__":
    main()
