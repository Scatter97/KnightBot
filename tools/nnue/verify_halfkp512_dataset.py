from __future__ import annotations

import argparse
import random
from pathlib import Path

import chess

from dataset_halfkp512 import (
    HalfKP512Store,
)

from halfkp_512 import (
    context_bits_from_board,
    halfkp_features_from_board,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--preencoded-dir",
        required=True,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    store = HalfKP512Store(
        args.preencoded_dir
    )

    rng = random.Random(
        args.seed
    )

    selected = sorted(
        rng.sample(
            range(
                store.sample_count
            ),
            min(
                args.samples,
                store.sample_count,
            ),
        )
    )

    selected_set = set(
        selected
    )

    verified = 0

    with Path(
        args.input
    ).open(
        "r",
        encoding="utf-8",
    ) as source:
        for line_index, line in enumerate(
            source
        ):
            if line_index > selected[-1]:
                break

            if line_index not in selected_set:
                continue

            fen, target_text = (
                line.rstrip(
                    "\r\n"
                ).rsplit(
                    "\t",
                    1,
                )
            )

            board = chess.Board(
                fen
            )

            expected_white = (
                halfkp_features_from_board(
                    board,
                    chess.WHITE,
                )
            )

            expected_black = (
                halfkp_features_from_board(
                    board,
                    chess.BLACK,
                )
            )

            expected_context = (
                context_bits_from_board(
                    board
                )
            )

            expected_target = int(
                target_text
            )

            (
                actual_white,
                actual_black,
                actual_context,
                actual_target,
            ) = store.sample(
                line_index
            )

            if (
                expected_white
                !=
                actual_white.astype(
                    "int64"
                ).tolist()
            ):
                raise RuntimeError(
                    f"White HalfKP mismatch "
                    f"at sample {line_index}"
                )

            if (
                expected_black
                !=
                actual_black.astype(
                    "int64"
                ).tolist()
            ):
                raise RuntimeError(
                    f"Black HalfKP mismatch "
                    f"at sample {line_index}"
                )

            if (
                expected_context
                !=
                actual_context
            ):
                raise RuntimeError(
                    f"Context mismatch "
                    f"at sample {line_index}"
                )

            if (
                expected_target
                !=
                actual_target
            ):
                raise RuntimeError(
                    f"Target mismatch "
                    f"at sample {line_index}"
                )

            verified += 1

    print()
    print(
        "# HalfKP-512 verification successful"
    )
    print()
    print(
        f"Samples checked:       {verified:,}"
    )
    print(
        "White perspective:     EXACT MATCH"
    )
    print(
        "Black perspective:     EXACT MATCH"
    )
    print(
        "Context features:      EXACT MATCH"
    )
    print(
        "Targets:               EXACT MATCH"
    )


if __name__ == "__main__":
    main()