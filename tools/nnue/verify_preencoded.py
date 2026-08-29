from __future__ import annotations

import argparse
import random
from pathlib import Path

from tqdm import tqdm

from dataset import (
    PreencodedStore,
)

from knightbot_nnue import (
    active_features_from_fen,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that a KnightBot preencoded dataset "
            "matches the original FEN/CP TSV exactly."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Original FEN<TAB>target_cp TSV.",
    )

    parser.add_argument(
        "--preencoded-dir",
        required=True,
        help="Preencoded dataset directory.",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help=(
            "Number of random samples to compare."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(
        args.input
    )

    store = PreencodedStore(
        args.preencoded_dir
    )

    if store.sample_count <= 0:
        raise RuntimeError(
            "Preencoded dataset is empty."
        )

    sample_count = min(
        args.samples,
        store.sample_count,
    )

    rng = random.Random(
        args.seed
    )

    selected_indices = sorted(
        rng.sample(
            range(
                store.sample_count
            ),
            sample_count,
        )
    )

    selected_set = set(
        selected_indices
    )

    maximum_index = (
        selected_indices[-1]
    )

    verified = 0

    print(
        f"Checking {sample_count:,} "
        f"random positions..."
    )

    with input_path.open(
        "r",
        encoding="utf-8",
        errors="strict",
    ) as source:
        progress = tqdm(
            total=maximum_index + 1,
            desc="Scanning TSV",
            unit="line",
        )

        try:
            for line_index, line in enumerate(
                source
            ):
                if line_index > maximum_index:
                    break

                progress.update(1)

                if line_index not in selected_set:
                    continue

                line = line.rstrip(
                    "\r\n"
                )

                if not line:
                    raise RuntimeError(
                        f"Blank TSV line at "
                        f"index {line_index}"
                    )

                fen, target_text = (
                    line.rsplit(
                        "\t",
                        1,
                    )
                )

                expected_target = int(
                    target_text
                )

                expected_features = (
                    active_features_from_fen(
                        fen
                    )
                )

                actual_features = (
                    store.features_for_sample(
                        line_index
                    )
                    .astype(
                        "int64"
                    )
                    .tolist()
                )

                actual_target = (
                    store.target_for_sample(
                        line_index
                    )
                )

                if (
                    expected_features
                    !=
                    actual_features
                ):
                    raise RuntimeError(
                        "Feature mismatch at sample "
                        f"{line_index}\n"
                        f"FEN: {fen}\n"
                        f"Expected: "
                        f"{expected_features}\n"
                        f"Actual:   "
                        f"{actual_features}"
                    )

                if (
                    expected_target
                    !=
                    actual_target
                ):
                    raise RuntimeError(
                        "Target mismatch at sample "
                        f"{line_index}: "
                        f"{expected_target} != "
                        f"{actual_target}"
                    )

                verified += 1

        finally:
            progress.close()

    if verified != sample_count:
        raise RuntimeError(
            "Could not verify all selected "
            f"samples: {verified} / "
            f"{sample_count}"
        )

    print()
    print(
        "# Verification successful"
    )
    print()
    print(
        f"Samples checked: {verified:,}"
    )
    print(
        "Feature mapping: EXACT MATCH"
    )
    print(
        "Targets:         EXACT MATCH"
    )
    print(
        "Preencoded dataset is compatible "
        "with KnightBot NNUE v1."
    )


if __name__ == "__main__":
    main()