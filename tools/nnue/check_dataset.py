from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import chess
from tqdm import tqdm

from knightbot_nnue import (
    normalized_position_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
    )

    parser.add_argument(
        "--duplicate-check-limit",
        type=int,
        default=2_000_000,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    path = Path(
        args.data
    )

    valid = 0
    invalid = 0
    duplicates = 0

    scores: list[int] = []

    seen: set[str] = set()

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as source:

        for line in tqdm(
            source,
            desc="Checking dataset",
        ):
            line = line.rstrip(
                "\r\n"
            )

            if not line:
                continue

            try:
                fen, score_text = (
                    line.rsplit(
                        "\t",
                        1,
                    )
                )

                board = chess.Board(
                    fen
                )

                score = int(
                    score_text
                )

            except (
                ValueError,
                IndexError,
            ):
                invalid += 1
                continue

            valid += 1

            scores.append(
                score
            )

            if (
                valid <=
                args.duplicate_check_limit
            ):
                key = (
                    normalized_position_key(
                        board
                    )
                )

                if key in seen:
                    duplicates += 1

                else:
                    seen.add(
                        key
                    )

    print(
        "\nDataset report"
    )

    print(
        "================================"
    )

    print(
        f"Valid positions:   {valid:,}"
    )

    print(
        f"Invalid lines:     {invalid:,}"
    )

    print(
        f"Duplicates found:  {duplicates:,}"
    )

    if scores:
        sorted_scores = sorted(
            scores
        )

        print(
            f"Minimum CP:        "
            f"{sorted_scores[0]}"
        )

        print(
            f"Maximum CP:        "
            f"{sorted_scores[-1]}"
        )

        print(
            f"Mean CP:           "
            f"{statistics.fmean(scores):.2f}"
        )

        print(
            f"Median CP:         "
            f"{statistics.median(scores):.2f}"
        )

        near_equal = sum(
            1
            for score in scores
            if abs(score) <= 50
        )

        decisive = sum(
            1
            for score in scores
            if abs(score) >= 500
        )

        print(
            f"|score| <= 50:     "
            f"{near_equal / len(scores):.1%}"
        )

        print(
            f"|score| >= 500:    "
            f"{decisive / len(scores):.1%}"
        )


if __name__ == "__main__":
    main()
