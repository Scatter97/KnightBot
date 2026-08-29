from __future__ import annotations

import argparse
import json
import os
import struct
import time
from array import array
from pathlib import Path

import chess
import numpy as np
from tqdm import tqdm

from dataset_halfkp512 import (
    BLACK_FEATURES_FILENAME,
    CONTEXT_FILENAME,
    FORMAT_VERSION,
    METADATA_FILENAME,
    OFFSETS_FILENAME,
    TARGETS_FILENAME,
    TRAIN_INDICES_FILENAME,
    VALIDATION_INDICES_FILENAME,
    WHITE_FEATURES_FILENAME,
)

from halfkp_512 import (
    HALFKP_FEATURE_COUNT,
    context_bits_from_board,
    halfkp_features_from_board,
)

from knightbot_nnue import (
    stable_split_value,
)


FLUSH_EVERY = 50000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preencode KnightBot HalfKP-512 training data."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--validation-permyriad",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--expected-count",
        type=int,
        default=0,
    )

    return parser.parse_args()


def flush_array(
    values: array,
    dtype: str,
    output,
) -> None:
    if not values:
        return

    np.asarray(
        values,
        dtype=dtype,
    ).tofile(
        output
    )

    del values[:]


def main() -> None:
    args = parse_args()

    input_path = Path(
        args.input
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not (
        0 <
        args.validation_permyriad <
        10000
    ):
        raise ValueError(
            "Invalid validation split."
        )

    paths = {
        "white": (
            output_dir /
            WHITE_FEATURES_FILENAME
        ),
        "black": (
            output_dir /
            BLACK_FEATURES_FILENAME
        ),
        "offsets": (
            output_dir /
            OFFSETS_FILENAME
        ),
        "context": (
            output_dir /
            CONTEXT_FILENAME
        ),
        "targets": (
            output_dir /
            TARGETS_FILENAME
        ),
        "train": (
            output_dir /
            TRAIN_INDICES_FILENAME
        ),
        "validation": (
            output_dir /
            VALIDATION_INDICES_FILENAME
        ),
        "meta": (
            output_dir /
            METADATA_FILENAME
        ),
    }

    temp = {
        key: Path(
            str(path) + ".tmp"
        )
        for key, path in paths.items()
    }

    for path in temp.values():
        if path.exists():
            path.unlink()

    white_buffer = array("H")
    black_buffer = array("H")
    offset_buffer = array("Q")
    context_buffer = array("H")
    target_buffer = array("h")
    train_buffer = array("I")
    validation_buffer = array("I")

    sample_count = 0
    flat_feature_count = 0
    train_count = 0
    validation_count = 0

    total = None

    if args.limit > 0:
        total = args.limit

    elif args.expected_count > 0:
        total = args.expected_count

    start_time = time.perf_counter()

    try:
        with (
            input_path.open(
                "r",
                encoding="utf-8",
                errors="strict",
            ) as source,
            temp["white"].open("wb") as white_output,
            temp["black"].open("wb") as black_output,
            temp["offsets"].open("wb") as offsets_output,
            temp["context"].open("wb") as context_output,
            temp["targets"].open("wb") as targets_output,
            temp["train"].open("wb") as train_output,
            temp["validation"].open("wb") as validation_output,
        ):
            offsets_output.write(
                struct.pack(
                    "<Q",
                    0,
                )
            )

            progress = tqdm(
                total=total,
                desc="HalfKP-512 preencoding",
                unit="pos",
            )

            for line_number, line in enumerate(
                source,
                start=1,
            ):
                if (
                    args.limit > 0
                    and
                    sample_count >= args.limit
                ):
                    break

                line = line.rstrip(
                    "\r\n"
                )

                fen, target_text = (
                    line.rsplit(
                        "\t",
                        1,
                    )
                )

                target = int(
                    target_text
                )

                board = chess.Board(
                    fen
                )

                white_features = (
                    halfkp_features_from_board(
                        board,
                        chess.WHITE,
                    )
                )

                black_features = (
                    halfkp_features_from_board(
                        board,
                        chess.BLACK,
                    )
                )

                if len(
                    white_features
                ) != len(
                    black_features
                ):
                    raise RuntimeError(
                        f"HalfKP perspective feature count "
                        f"mismatch on line {line_number}"
                    )

                white_buffer.extend(
                    white_features
                )

                black_buffer.extend(
                    black_features
                )

                flat_feature_count += len(
                    white_features
                )

                offset_buffer.append(
                    flat_feature_count
                )

                context_buffer.append(
                    context_bits_from_board(
                        board
                    )
                )

                target_buffer.append(
                    target
                )

                split_value = (
                    stable_split_value(
                        fen
                    )
                    %
                    10000
                )

                if (
                    split_value <
                    args.validation_permyriad
                ):
                    validation_buffer.append(
                        sample_count
                    )

                    validation_count += 1

                else:
                    train_buffer.append(
                        sample_count
                    )

                    train_count += 1

                sample_count += 1
                progress.update(1)

                if (
                    sample_count %
                    FLUSH_EVERY
                    ==
                    0
                ):
                    flush_array(
                        white_buffer,
                        "<u2",
                        white_output,
                    )

                    flush_array(
                        black_buffer,
                        "<u2",
                        black_output,
                    )

                    flush_array(
                        offset_buffer,
                        "<u8",
                        offsets_output,
                    )

                    flush_array(
                        context_buffer,
                        "<u2",
                        context_output,
                    )

                    flush_array(
                        target_buffer,
                        "<i2",
                        targets_output,
                    )

                    flush_array(
                        train_buffer,
                        "<u4",
                        train_output,
                    )

                    flush_array(
                        validation_buffer,
                        "<u4",
                        validation_output,
                    )

            progress.close()

            flush_array(
                white_buffer,
                "<u2",
                white_output,
            )

            flush_array(
                black_buffer,
                "<u2",
                black_output,
            )

            flush_array(
                offset_buffer,
                "<u8",
                offsets_output,
            )

            flush_array(
                context_buffer,
                "<u2",
                context_output,
            )

            flush_array(
                target_buffer,
                "<i2",
                targets_output,
            )

            flush_array(
                train_buffer,
                "<u4",
                train_output,
            )

            flush_array(
                validation_buffer,
                "<u4",
                validation_output,
            )

        if (
            args.expected_count > 0
            and
            args.limit <= 0
            and
            sample_count !=
            args.expected_count
        ):
            raise RuntimeError(
                f"Expected {args.expected_count:,} "
                f"positions, got {sample_count:,}."
            )

        metadata = {
            "format_version": (
                FORMAT_VERSION
            ),
            "architecture": (
                "HalfKP-512"
            ),
            "feature_count": (
                HALFKP_FEATURE_COUNT
            ),
            "sample_count": (
                sample_count
            ),
            "flat_feature_count": (
                flat_feature_count
            ),
            "train_count": (
                train_count
            ),
            "validation_count": (
                validation_count
            ),
            "validation_permyriad": (
                args.validation_permyriad
            ),
            "source": str(
                input_path.resolve()
            ).replace(
                "\\",
                "/",
            ),
        }

        temp[
            "meta"
        ].write_text(
            json.dumps(
                metadata,
                indent=2,
            )
            +
            "\n",
            encoding="utf-8",
        )

        for key in (
            "white",
            "black",
            "offsets",
            "context",
            "targets",
            "train",
            "validation",
            "meta",
        ):
            os.replace(
                temp[key],
                paths[key],
            )

    except Exception:
        for path in temp.values():
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass

        raise

    elapsed = (
        time.perf_counter() -
        start_time
    )

    total_bytes = sum(
        path.stat().st_size
        for key, path in paths.items()
        if key != "meta"
    )

    print()
    print(
        "# HalfKP-512 preencoding complete"
    )
    print()
    print(
        f"Positions:          "
        f"{sample_count:,}"
    )
    print(
        f"Features/persp.:    "
        f"{flat_feature_count:,}"
    )
    print(
        f"Train:              "
        f"{train_count:,}"
    )
    print(
        f"Validation:         "
        f"{validation_count:,}"
    )
    print(
        f"Elapsed:            "
        f"{elapsed:.1f} sec"
    )
    print(
        f"Throughput:         "
        f"{sample_count / elapsed:,.0f} pos/s"
    )
    print(
        f"Binary size:        "
        f"{total_bytes / (1024 ** 2):,.1f} MiB"
    )
    print(
        f"Output:             "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()