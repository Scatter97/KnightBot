from __future__ import annotations

import argparse
import random
from pathlib import Path

import chess
import numpy as np
import torch

from dataset_halfkp512 import HalfKP512Store

from halfkp_512 import (
    CONTEXT_COUNT,
    HalfKP512,
    evaluate_quantized_board,
    load_quantized_network,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare HalfKP-512 float checkpoint "
            "against quantized NNUE inference."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="HalfKP-512 PyTorch checkpoint.",
    )

    parser.add_argument(
        "--network",
        required=True,
        help="Quantized HalfKP-512 .nnue file.",
    )

    parser.add_argument(
        "--data",
        required=True,
        help="HalfKP-512 preencoded dataset directory.",
    )

    parser.add_argument(
        "--source-tsv",
        required=True,
        help="Original labeled TSV used for preencoding.",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of validation positions to test.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    return parser.parse_args()


def load_selected_fens(
    source_tsv: Path,
    selected_sample_indices: list[int],
) -> dict[int, str]:
    """
    Read the 5M-line TSV only once.

    selected_sample_indices are absolute sample indices
    in the original TSV/preencoded store.
    """

    wanted = set(
        selected_sample_indices
    )

    result: dict[int, str] = {}

    highest_needed = max(
        selected_sample_indices
    )

    with source_tsv.open(
        "r",
        encoding="utf-8",
        errors="strict",
    ) as source:
        for line_index, line in enumerate(
            source
        ):
            if line_index > highest_needed:
                break

            if line_index not in wanted:
                continue

            line = line.rstrip(
                "\r\n"
            )

            fen, _target = line.rsplit(
                "\t",
                1,
            )

            result[
                line_index
            ] = fen

            if len(result) == len(wanted):
                break

    missing = (
        wanted -
        result.keys()
    )

    if missing:
        preview = sorted(
            missing
        )[:10]

        raise RuntimeError(
            "Could not find all selected positions "
            f"in source TSV. Missing examples: {preview}"
        )

    return result


def make_float_inputs(
    store: HalfKP512Store,
    sample_index: int,
    device: torch.device,
):
    (
        white_features,
        black_features,
        context_bits,
        _target,
    ) = store.sample(
        sample_index
    )

    white = torch.from_numpy(
        white_features.astype(
            np.int64,
            copy=True,
        )
    ).to(
        device,
        non_blocking=True,
    )

    black = torch.from_numpy(
        black_features.astype(
            np.int64,
            copy=True,
        )
    ).to(
        device,
        non_blocking=True,
    )

    offsets = torch.zeros(
        1,
        dtype=torch.long,
        device=device,
    )

    context = torch.zeros(
        (
            1,
            CONTEXT_COUNT,
        ),
        dtype=torch.float32,
        device=device,
    )

    for index in range(
        CONTEXT_COUNT
    ):
        if context_bits & (
            1 << index
        ):
            context[
                0,
                index,
            ] = 1.0

    return (
        white,
        black,
        offsets,
        context,
    )


def percentile(
    values: np.ndarray,
    q: float,
) -> float:
    return float(
        np.percentile(
            values,
            q,
        )
    )


def main() -> None:
    args = parse_args()

    if args.samples <= 0:
        raise ValueError(
            "--samples must be greater than zero."
        )

    source_tsv = Path(
        args.source_tsv
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    print(
        f"Device:              {device}"
    )

    if device.type == "cuda":
        print(
            "CUDA device:         "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Load float checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model = HalfKP512().to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ]
    )

    model.eval()

    checkpoint_epoch = int(
        checkpoint[
            "epoch"
        ]
    )

    # --------------------------------------------------------
    # Load quantized network + dataset
    # --------------------------------------------------------

    quantized = load_quantized_network(
        args.network
    )

    store = HalfKP512Store(
        args.data
    )

    validation_indices = np.memmap(
        store.validation_indices_path,
        dtype="<u4",
        mode="r",
    )

    if len(
        validation_indices
    ) != store.validation_count:
        raise RuntimeError(
            "Validation index count does not "
            "match dataset metadata."
        )

    sample_count = min(
        args.samples,
        store.validation_count,
    )

    rng = random.Random(
        args.seed
    )

    selected_validation_slots = (
        rng.sample(
            range(
                store.validation_count
            ),
            sample_count,
        )
    )

    selected_sample_indices = [
        int(
            validation_indices[
                slot
            ]
        )
        for slot in selected_validation_slots
    ]

    # Sort by original TSV position so loading the FENs
    # requires only one sequential pass through the TSV.
    selected_sample_indices.sort()

    print(
        f"Checkpoint epoch:    {checkpoint_epoch}"
    )
    print(
        f"Validation positions:{store.validation_count:>12,}"
    )
    print(
        f"Testing positions:   {sample_count:>12,}"
    )
    print()
    print(
        "Reading selected FENs from TSV..."
    )

    fens = load_selected_fens(
        source_tsv,
        selected_sample_indices,
    )

    print(
        "Running float vs quantized comparison..."
    )
    print()

    differences = np.empty(
        sample_count,
        dtype=np.float64,
    )

    signed_differences = np.empty(
        sample_count,
        dtype=np.float64,
    )

    float_scores = np.empty(
        sample_count,
        dtype=np.float64,
    )

    quantized_scores = np.empty(
        sample_count,
        dtype=np.int32,
    )

    worst_index = -1
    worst_difference = -1.0

    with torch.no_grad():
        for number, sample_index in enumerate(
            selected_sample_indices,
            start=1,
        ):
            (
                white,
                black,
                offsets,
                context,
            ) = make_float_inputs(
                store,
                sample_index,
                device,
            )

            float_score = float(
                model(
                    white,
                    black,
                    offsets,
                    context,
                ).item()
            )

            board = chess.Board(
                fens[
                    sample_index
                ]
            )

            quantized_score = int(
                evaluate_quantized_board(
                    quantized,
                    board,
                )
            )

            signed_difference = (
                quantized_score -
                float_score
            )

            absolute_difference = abs(
                signed_difference
            )

            array_index = (
                number - 1
            )

            float_scores[
                array_index
            ] = float_score

            quantized_scores[
                array_index
            ] = quantized_score

            signed_differences[
                array_index
            ] = signed_difference

            differences[
                array_index
            ] = absolute_difference

            if (
                absolute_difference >
                worst_difference
            ):
                worst_difference = (
                    absolute_difference
                )

                worst_index = (
                    array_index
                )

            if (
                number % 100 == 0
                or
                number == sample_count
            ):
                print(
                    f"{number:,}/"
                    f"{sample_count:,}"
                )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    mean_absolute = float(
        np.mean(
            differences
        )
    )

    mean_signed = float(
        np.mean(
            signed_differences
        )
    )

    median_absolute = float(
        np.median(
            differences
        )
    )

    maximum_absolute = float(
        np.max(
            differences
        )
    )

    within_1 = float(
        np.mean(
            differences <= 1.0
        )
        *
        100.0
    )

    within_2 = float(
        np.mean(
            differences <= 2.0
        )
        *
        100.0
    )

    within_5 = float(
        np.mean(
            differences <= 5.0
        )
        *
        100.0
    )

    within_10 = float(
        np.mean(
            differences <= 10.0
        )
        *
        100.0
    )

    print()
    print(
        "# HalfKP-512 quantization report"
    )
    print()

    print(
        f"Samples:                 "
        f"{sample_count:,}"
    )

    print(
        f"Mean absolute error:     "
        f"{mean_absolute:.3f} cp"
    )

    print(
        f"Median absolute error:   "
        f"{median_absolute:.3f} cp"
    )

    print(
        f"Mean signed error:       "
        f"{mean_signed:+.3f} cp"
    )

    print(
        f"95th percentile:         "
        f"{percentile(differences, 95):.3f} cp"
    )

    print(
        f"99th percentile:         "
        f"{percentile(differences, 99):.3f} cp"
    )

    print(
        f"Maximum absolute error:  "
        f"{maximum_absolute:.3f} cp"
    )

    print()

    print(
        f"Within 1 cp:             "
        f"{within_1:.1f}%"
    )

    print(
        f"Within 2 cp:             "
        f"{within_2:.1f}%"
    )

    print(
        f"Within 5 cp:             "
        f"{within_5:.1f}%"
    )

    print(
        f"Within 10 cp:            "
        f"{within_10:.1f}%"
    )

    print()

    print(
        "# Worst sampled position"
    )
    print()

    worst_sample_index = (
        selected_sample_indices[
            worst_index
        ]
    )

    print(
        f"Dataset index:           "
        f"{worst_sample_index:,}"
    )

    print(
        f"Float score:             "
        f"{float_scores[worst_index]:.3f} cp"
    )

    print(
        f"Quantized score:         "
        f"{quantized_scores[worst_index]} cp"
    )

    print(
        f"Difference:              "
        f"{signed_differences[worst_index]:+.3f} cp"
    )

    print(
        f"FEN:"
    )

    print(
        fens[
            worst_sample_index
        ]
    )


if __name__ == "__main__":
    main()