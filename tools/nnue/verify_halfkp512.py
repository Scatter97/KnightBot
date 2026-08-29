from __future__ import annotations

import argparse

from halfkp_512 import (
    evaluate_quantized_fen,
    load_quantized_network,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--network",
        required=True,
    )

    parser.add_argument(
        "--fen",
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    network = load_quantized_network(
        args.network
    )

    score = evaluate_quantized_fen(
        network,
        args.fen,
    )

    print(
        score
    )


if __name__ == "__main__":
    main()