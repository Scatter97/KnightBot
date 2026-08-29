from __future__ import annotations

import argparse

from knightbot_nnue import (
    evaluate_quantized_fen,
    load_quantized_network,
)


DEFAULT_TEST_POSITIONS = [
    (
        "startpos",
        "rnbqkbnr/pppppppp/8/8/8/8/"
        "PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ),

    (
        "white up pawn",
        "rnbqkbnr/ppp1pppp/8/3P4/"
        "8/8/PPPP1PPP/RNBQKBNR "
        "b KQkq - 0 2",
    ),

    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/"
        "3PN3/1p2P3/2N2Q1p/"
        "PPPBBPPP/R3K2R "
        "w KQkq - 0 1",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--network",
        required=True,
    )

    parser.add_argument(
        "--fen",
        default="",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    network = load_quantized_network(
        args.network
    )

    if args.fen:
        score = evaluate_quantized_fen(
            network,
            args.fen,
        )

        print(
            score
        )

        return

    print(
        "KnightBot integer NNUE verification"
    )

    print(
        "==================================="
    )

    for name, fen in DEFAULT_TEST_POSITIONS:
        score = evaluate_quantized_fen(
            network,
            fen,
        )

        print(
            f"{name:20s} "
            f"{score:+d} cp"
        )


if __name__ == "__main__":
    main()
