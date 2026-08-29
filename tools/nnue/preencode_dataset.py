from __future__ import annotations

import argparse
import json
import os
import struct
import time
from array import array
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dataset import (
    FEATURES_FILENAME,
    METADATA_FILENAME,
    OFFSETS_FILENAME,
    PREENCODED_FORMAT_VERSION,
    TARGETS_FILENAME,
    TRAIN_INDICES_FILENAME,
    VALIDATION_INDICES_FILENAME,
)

from knightbot_nnue import (
    INPUT_COUNT,
    active_features_from_fen,
    stable_split_value,
)


FLUSH_SAMPLE_COUNT = 50000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preencode KnightBot FEN/CP TSV data into "
            "memory-mapped sparse NNUE training files."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input FEN<TAB>target_cp TSV.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the binary dataset.",
    )

    parser.add_argument(
        "--validation-permyriad",
        type=int,
        default=500,
        help=(
            "Validation split. "
            "500 means 5 percent."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Process only this many positions. "
            "0 means all positions."
        ),
    )

    parser.add_argument(
        "--expected-count",
        type=int,
        default=0,
        help=(
            "Optional expected number of positions, "
            "used for the progress bar and validation."
        ),
    )

    return parser.parse_args()


def flush_uint16(
    values: array,
    output,
) -> None:
    if not values:
        return

    np.asarray(
        values,
        dtype="<u2",
    ).tofile(
        output
    )

    del values[:]


def flush_int16(
    values: array,
    output,
) -> None:
    if not values:
        return

    np.asarray(
        values,
        dtype="<i2",
    ).tofile(
        output
    )

    del values[:]


def flush_uint32(
    values: array,
    output,
) -> None:
    if not values:
        return

    np.asarray(
        values,
        dtype="<u4",
    ).tofile(
        output
    )

    del values[:]


def flush_uint64(
    values: array,
    output,
) -> None:
    if not values:
        return

    np.asarray(
        values,
        dtype="<u8",
    ).tofile(
        output
    )

    del values[:]


def main() -> None:
    args = parse_args()

    if not (
        0 <
        args.validation_permyriad <
        10000
    ):
        raise ValueError(
            "--validation-permyriad must be "
            "between 1 and 9999"
        )

    input_path = Path(
        args.input
    )

    output_dir = Path(
        args.output_dir
    )

    if not input_path.exists():
        raise FileNotFoundError(
            input_path
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_paths = {
        "features": (
            output_dir /
            FEATURES_FILENAME
        ),
        "offsets": (
            output_dir /
            OFFSETS_FILENAME
        ),
        "targets": (
            output_dir /
            TARGETS_FILENAME
        ),
        "train_indices": (
            output_dir /
            TRAIN_INDICES_FILENAME
        ),
        "validation_indices": (
            output_dir /
            VALIDATION_INDICES_FILENAME
        ),
        "metadata": (
            output_dir /
            METADATA_FILENAME
        ),
    }

    temp_paths = {
        name: Path(
            str(path) + ".tmp"
        )
        for name, path in final_paths.items()
    }

    for path in temp_paths.values():
        if path.exists():
            path.unlink()

    total_for_progress: int | None

    if args.limit > 0:
        total_for_progress = (
            args.limit
        )
    elif args.expected_count > 0:
        total_for_progress = (
            args.expected_count
        )
    else:
        total_for_progress = None

    feature_buffer = array("H")
    offset_buffer = array("Q")
    target_buffer = array("h")
    train_index_buffer = array("I")
    validation_index_buffer = array("I")

    sample_count = 0
    feature_count = 0
    train_count = 0
    validation_count = 0

    start_time = time.perf_counter()

    try:
        with (
            input_path.open(
                "r",
                encoding="utf-8",
                errors="strict",
            ) as source,
            temp_paths[
                "features"
            ].open("wb") as features_output,
            temp_paths[
                "offsets"
            ].open("wb") as offsets_output,
            temp_paths[
                "targets"
            ].open("wb") as targets_output,
            temp_paths[
                "train_indices"
            ].open("wb") as train_indices_output,
            temp_paths[
                "validation_indices"
            ].open("wb") as validation_indices_output,
        ):
            # offsets.bin contains N + 1 entries.
            # The first sample always begins at feature offset zero.
            offsets_output.write(
                struct.pack(
                    "<Q",
                    0,
                )
            )

            progress = tqdm(
                total=total_for_progress,
                desc="Preencoding",
                unit="pos",
            )

            try:
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

                    if not line:
                        raise ValueError(
                            "Blank line encountered at "
                            f"input line {line_number}"
                        )

                    try:
                        fen, target_text = (
                            line.rsplit(
                                "\t",
                                1,
                            )
                        )

                        target = int(
                            target_text
                        )

                    except (
                        ValueError,
                        IndexError,
                    ) as error:
                        raise ValueError(
                            "Invalid TSV line at "
                            f"{line_number}: {line!r}"
                        ) from error

                    if not (
                        np.iinfo(
                            np.int16
                        ).min
                        <=
                        target
                        <=
                        np.iinfo(
                            np.int16
                        ).max
                    ):
                        raise ValueError(
                            "Target does not fit int16 at "
                            f"line {line_number}: {target}"
                        )

                    try:
                        features = (
                            active_features_from_fen(
                                fen
                            )
                        )
                    except Exception as error:
                        raise ValueError(
                            "Invalid FEN at input line "
                            f"{line_number}: {fen}"
                        ) from error

                    for feature in features:
                        if not (
                            0 <=
                            feature <
                            INPUT_COUNT
                        ):
                            raise ValueError(
                                "Feature index out of range "
                                f"at line {line_number}: "
                                f"{feature}"
                            )

                    feature_buffer.extend(
                        features
                    )

                    feature_count += len(
                        features
                    )

                    offset_buffer.append(
                        feature_count
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
                        validation_index_buffer.append(
                            sample_count
                        )

                        validation_count += 1

                    else:
                        train_index_buffer.append(
                            sample_count
                        )

                        train_count += 1

                    sample_count += 1
                    progress.update(1)

                    if (
                        sample_count %
                        FLUSH_SAMPLE_COUNT
                        ==
                        0
                    ):
                        flush_uint16(
                            feature_buffer,
                            features_output,
                        )

                        flush_uint64(
                            offset_buffer,
                            offsets_output,
                        )

                        flush_int16(
                            target_buffer,
                            targets_output,
                        )

                        flush_uint32(
                            train_index_buffer,
                            train_indices_output,
                        )

                        flush_uint32(
                            validation_index_buffer,
                            validation_indices_output,
                        )

            finally:
                progress.close()

            flush_uint16(
                feature_buffer,
                features_output,
            )

            flush_uint64(
                offset_buffer,
                offsets_output,
            )

            flush_int16(
                target_buffer,
                targets_output,
            )

            flush_uint32(
                train_index_buffer,
                train_indices_output,
            )

            flush_uint32(
                validation_index_buffer,
                validation_indices_output,
            )

            features_output.flush()
            offsets_output.flush()
            targets_output.flush()
            train_indices_output.flush()
            validation_indices_output.flush()

            os.fsync(
                features_output.fileno()
            )
            os.fsync(
                offsets_output.fileno()
            )
            os.fsync(
                targets_output.fileno()
            )
            os.fsync(
                train_indices_output.fileno()
            )
            os.fsync(
                validation_indices_output.fileno()
            )

        if (
            args.expected_count > 0
            and
            args.limit <= 0
            and
            sample_count != args.expected_count
        ):
            raise RuntimeError(
                "Processed position count does not "
                "match --expected-count: "
                f"{sample_count:,} != "
                f"{args.expected_count:,}"
            )

        metadata = {
            "format_version": (
                PREENCODED_FORMAT_VERSION
            ),
            "input_count": (
                INPUT_COUNT
            ),
            "sample_count": (
                sample_count
            ),
            "feature_count": (
                feature_count
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
            "feature_dtype": (
                "uint16"
            ),
            "offset_dtype": (
                "uint64"
            ),
            "target_dtype": (
                "int16"
            ),
            "index_dtype": (
                "uint32"
            ),
            "source": (
                str(
                    input_path.resolve()
                )
            ),
        }

        with temp_paths[
            "metadata"
        ].open(
            "w",
            encoding="utf-8",
        ) as output:
            json.dump(
                metadata,
                output,
                indent=2,
            )

            output.write(
                "\n"
            )

            output.flush()

            os.fsync(
                output.fileno()
            )

        # Only expose the finished dataset after every temporary
        # output file has completed successfully.
        for name in (
            "features",
            "offsets",
            "targets",
            "train_indices",
            "validation_indices",
            "metadata",
        ):
            os.replace(
                temp_paths[name],
                final_paths[name],
            )

    except Exception:
        print(
            "\nPreencoding failed. "
            "Removing temporary files..."
        )

        for path in temp_paths.values():
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

    positions_per_second = (
        sample_count /
        elapsed
        if elapsed > 0
        else 0.0
    )

    dataset_bytes = sum(
        path.stat().st_size
        for name, path
        in final_paths.items()
        if name != "metadata"
    )

    print()
    print(
        "# Preencoding complete"
    )
    print()
    print(
        f"Positions:       "
        f"{sample_count:,}"
    )
    print(
        f"Active features: "
        f"{feature_count:,}"
    )
    print(
        f"Train:           "
        f"{train_count:,}"
    )
    print(
        f"Validation:      "
        f"{validation_count:,}"
    )
    print(
        f"Elapsed:         "
        f"{elapsed:.1f} sec"
    )
    print(
        f"Throughput:      "
        f"{positions_per_second:,.0f} pos/s"
    )
    print(
        f"Binary size:     "
        f"{dataset_bytes / (1024 ** 2):,.1f} MiB"
    )
    print(
        f"Output:          "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()