from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    KnightBotDataset,
    PreencodedKnightBotDataset,
    PreencodedStore,
    collate_sparse,
)

from knightbot_nnue import (
    KnightBotNNUE,
    quantize_model,
    save_quantized_network,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train KnightBot NNUE v1."
        )
    )

    input_group = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    input_group.add_argument(
        "--data",
        help=(
            "Original Stockfish-labeled "
            "FEN/CP training TSV."
        ),
    )

    input_group.add_argument(
        "--preencoded-dir",
        help=(
            "Preencoded binary dataset directory. "
            "Recommended for large training runs."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="runs/knightbot-v1",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--validation-permyriad",
        type=int,
        default=500,
        help=(
            "500 = 5 percent validation. "
            "For preencoded data, this must "
            "match the preprocessing split."
        ),
    )

    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=25000,
        help=(
            "Only used with the original TSV dataset."
        ),
    )

    parser.add_argument(
        "--target-clamp",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--resume",
        default="",
        help="Checkpoint path.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
        help=(
            "Disable CUDA mixed precision."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help=(
            "PyTorch DataLoader worker processes."
        ),
    )

    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help=(
            "Batches prefetched by each DataLoader "
            "worker."
        ),
    )

    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep DataLoader workers alive "
            "between epochs."
        ),
    )

    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use pinned host memory for faster "
            "CUDA transfers."
        ),
    )

    parser.add_argument(
        "--benchmark-loader",
        action="store_true",
        help=(
            "Benchmark DataLoader throughput and exit."
        ),
    )

    parser.add_argument(
        "--benchmark-batches",
        type=int,
        default=100,
        help=(
            "Number of batches used by "
            "--benchmark-loader."
        ),
    )

    return parser.parse_args()


def make_loader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "collate_fn": collate_sparse,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if (
        shuffle
        and
        not isinstance(
            dataset,
            KnightBotDataset,
        )
    ):
        generator = torch.Generator()

        generator.manual_seed(
            seed
        )

        kwargs[
            "shuffle"
        ] = True

        kwargs[
            "generator"
        ] = generator

    if num_workers > 0:
        kwargs[
            "prefetch_factor"
        ] = prefetch_factor

        kwargs[
            "persistent_workers"
        ] = persistent_workers

    return DataLoader(
        **kwargs
    )


def move_batch(
    batch,
    device: torch.device,
):
    features, offsets, targets = batch

    return (
        features.to(
            device,
            non_blocking=True,
        ),
        offsets.to(
            device,
            non_blocking=True,
        ),
        targets.to(
            device,
            non_blocking=True,
        ),
    )


def save_checkpoint(
    path: Path,
    model: KnightBotNNUE,
    optimizer: AdamW,
    epoch: int,
    best_validation_loss: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": (
                model.state_dict()
            ),
            "optimizer_state": (
                optimizer.state_dict()
            ),
            "best_validation_loss": (
                best_validation_loss
            ),
        },
        path,
    )


def benchmark_loader(
    loader: DataLoader,
    max_batches: int,
) -> None:
    print()
    print(
        "# DataLoader benchmark"
    )
    print()

    start = time.perf_counter()

    sample_count = 0
    batch_count = 0
    feature_count = 0

    progress = tqdm(
        loader,
        desc="Loader benchmark",
    )

    try:
        for batch_index, batch in enumerate(
            progress
        ):
            if (
                max_batches > 0
                and
                batch_index >= max_batches
            ):
                break

            features, _, targets = batch

            sample_count += int(
                targets.numel()
            )

            feature_count += int(
                features.numel()
            )

            batch_count += 1

            elapsed = (
                time.perf_counter() -
                start
            )

            if elapsed > 0:
                progress.set_postfix(
                    samples_s=(
                        f"{sample_count / elapsed:,.0f}"
                    )
                )

    finally:
        progress.close()

    elapsed = (
        time.perf_counter() -
        start
    )

    if elapsed <= 0:
        elapsed = 1e-9

    print()
    print(
        f"Batches:          "
        f"{batch_count:,}"
    )
    print(
        f"Samples:          "
        f"{sample_count:,}"
    )
    print(
        f"Sparse features:  "
        f"{feature_count:,}"
    )
    print(
        f"Elapsed:          "
        f"{elapsed:.3f} sec"
    )
    print(
        f"Batches/sec:      "
        f"{batch_count / elapsed:,.2f}"
    )
    print(
        f"Samples/sec:      "
        f"{sample_count / elapsed:,.0f}"
    )

    if sample_count > 0:
        print(
            f"Features/sample:  "
            f"{feature_count / sample_count:.2f}"
        )


@torch.no_grad()
def validate(
    model: KnightBotNNUE,
    loader: DataLoader,
    device: torch.device,
    loss_function: nn.Module,
    max_batches: int,
    use_amp: bool,
) -> tuple[
    float,
    float,
    int,
    float,
]:
    model.eval()

    total_loss = 0.0
    total_absolute_error = 0.0
    sample_count = 0

    start = time.perf_counter()

    for batch_index, batch in enumerate(
        tqdm(
            loader,
            desc="Validation",
            leave=False,
        )
    ):
        if (
            max_batches > 0
            and
            batch_index >= max_batches
        ):
            break

        (
            features,
            offsets,
            targets,
        ) = move_batch(
            batch,
            device,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            predictions = model(
                features,
                offsets,
            )

            loss = loss_function(
                predictions,
                targets,
            )

        batch_size = (
            targets.numel()
        )

        total_loss += (
            float(
                loss.item()
            )
            *
            batch_size
        )

        total_absolute_error += float(
            torch.sum(
                torch.abs(
                    predictions.float()
                    -
                    targets
                )
            ).item()
        )

        sample_count += (
            batch_size
        )

    elapsed = (
        time.perf_counter() -
        start
    )

    if sample_count == 0:
        return (
            math.inf,
            math.inf,
            0,
            elapsed,
        )

    return (
        total_loss /
        sample_count,
        total_absolute_error /
        sample_count,
        sample_count,
        elapsed,
    )


def main() -> None:
    args = parse_args()

    if args.num_workers < 0:
        raise ValueError(
            "--num-workers cannot be negative"
        )

    if args.prefetch_factor < 1:
        raise ValueError(
            "--prefetch-factor must be >= 1"
        )

    torch.manual_seed(
        args.seed
    )

    if torch.cuda.is_available():
        device = torch.device(
            "cuda"
        )

        print(
            "CUDA device:",
            torch.cuda.get_device_name(0),
        )

    else:
        device = torch.device(
            "cpu"
        )

        print(
            "WARNING: CUDA not found. "
            "Training will run on CPU."
        )

    use_amp = (
        device.type == "cuda"
        and
        not args.no_amp
    )

    pin_memory = (
        args.pin_memory
        and
        device.type == "cuda"
    )

    print(
        f"AMP:              "
        f"{'on' if use_amp else 'off'}"
    )

    print(
        f"DataLoader workers:"
        f" {args.num_workers}"
    )

    print(
        f"Pin memory:       "
        f"{'on' if pin_memory else 'off'}"
    )

    if args.preencoded_dir:
        store = PreencodedStore(
            args.preencoded_dir
        )

        if (
            store.validation_permyriad
            !=
            args.validation_permyriad
        ):
            raise ValueError(
                "Preencoded validation split does "
                "not match --validation-permyriad: "
                f"{store.validation_permyriad} != "
                f"{args.validation_permyriad}"
            )

        train_dataset = (
            PreencodedKnightBotDataset(
                args.preencoded_dir,
                split="train",
                target_clamp=(
                    args.target_clamp
                ),
            )
        )

        validation_dataset = (
            PreencodedKnightBotDataset(
                args.preencoded_dir,
                split="validation",
                target_clamp=(
                    args.target_clamp
                ),
            )
        )

        print(
            "Dataset mode:     preencoded"
        )

        print(
            f"Total positions:  "
            f"{store.sample_count:,}"
        )

        print(
            f"Train positions:  "
            f"{store.train_count:,}"
        )

        print(
            f"Validation:       "
            f"{store.validation_count:,}"
        )

        train_shuffle = True

    else:
        train_dataset = (
            KnightBotDataset(
                args.data,
                split="train",
                validation_permyriad=(
                    args.validation_permyriad
                ),
                shuffle_buffer=(
                    args.shuffle_buffer
                ),
                seed=args.seed,
                target_clamp=(
                    args.target_clamp
                ),
            )
        )

        validation_dataset = (
            KnightBotDataset(
                args.data,
                split="validation",
                validation_permyriad=(
                    args.validation_permyriad
                ),
                shuffle_buffer=0,
                seed=args.seed,
                target_clamp=(
                    args.target_clamp
                ),
            )
        )

        print(
            "Dataset mode:     TSV/FEN"
        )

        train_shuffle = False

    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        prefetch_factor=(
            args.prefetch_factor
        ),
        persistent_workers=(
            args.persistent_workers
        ),
        shuffle=train_shuffle,
        seed=args.seed,
    )

    validation_loader = make_loader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        prefetch_factor=(
            args.prefetch_factor
        ),
        persistent_workers=(
            args.persistent_workers
        ),
        shuffle=False,
        seed=args.seed,
    )

    if args.benchmark_loader:
        benchmark_loader(
            train_loader,
            args.benchmark_batches,
        )

        return

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = KnightBotNNUE().to(
        device
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # CP labels contain occasional large tactical swings.
    # SmoothL1 is considerably more robust than pure MSE.
    loss_function = nn.SmoothL1Loss(
        beta=100.0
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    start_epoch = 1
    best_validation_loss = math.inf

    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint[
                "model_state"
            ]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state"
            ]
        )

        start_epoch = (
            int(
                checkpoint[
                    "epoch"
                ]
            )
            +
            1
        )

        best_validation_loss = float(
            checkpoint.get(
                "best_validation_loss",
                math.inf,
            )
        )

        print(
            f"Resuming at epoch "
            f"{start_epoch}"
        )

    metrics_path = (
        output_dir /
        "metrics.csv"
    )

    metrics_exists = (
        metrics_path.exists()
        and
        bool(args.resume)
    )

    metrics_file = (
        metrics_path.open(
            "a"
            if metrics_exists
            else
            "w",
            encoding="utf-8",
            newline="",
        )
    )

    metrics_writer = csv.writer(
        metrics_file
    )

    if not metrics_exists:
        metrics_writer.writerow(
            [
                "epoch",
                "train_loss",
                "validation_loss",
                "validation_mae_cp",
                "seconds",
                "train_samples",
                "train_samples_per_sec",
                "train_batches_per_sec",
                "validation_samples",
                "validation_seconds",
            ]
        )

        metrics_file.flush()

    try:
        for epoch in range(
            start_epoch,
            args.epochs + 1,
        ):
            print()
            print(
                f"Epoch {epoch}/"
                f"{args.epochs}"
            )

            model.train()

            epoch_start = (
                time.perf_counter()
            )

            running_loss = 0.0
            sample_count = 0
            batch_count = 0

            progress = tqdm(
                train_loader,
                desc="Training",
            )

            for (
                batch_index,
                batch,
            ) in enumerate(
                progress
            ):
                if (
                    args.max_train_batches > 0
                    and
                    batch_index >=
                    args.max_train_batches
                ):
                    break

                (
                    features,
                    offsets,
                    targets,
                ) = move_batch(
                    batch,
                    device,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    predictions = model(
                        features,
                        offsets,
                    )

                    loss = (
                        loss_function(
                            predictions,
                            targets,
                        )
                    )

                scaler.scale(
                    loss
                ).backward()

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                batch_size = (
                    targets.numel()
                )

                running_loss += (
                    float(
                        loss.item()
                    )
                    *
                    batch_size
                )

                sample_count += (
                    batch_size
                )

                batch_count += 1

                average_loss = (
                    running_loss /
                    max(
                        1,
                        sample_count,
                    )
                )

                train_elapsed = (
                    time.perf_counter()
                    -
                    epoch_start
                )

                samples_per_second = (
                    sample_count /
                    train_elapsed
                    if train_elapsed > 0
                    else 0.0
                )

                progress.set_postfix(
                    loss=(
                        f"{average_loss:.4f}"
                    ),
                    samples_s=(
                        f"{samples_per_second:,.0f}"
                    ),
                )

            train_elapsed = (
                time.perf_counter()
                -
                epoch_start
            )

            train_loss = (
                running_loss /
                max(
                    1,
                    sample_count,
                )
            )

            train_samples_per_second = (
                sample_count /
                train_elapsed
                if train_elapsed > 0
                else 0.0
            )

            train_batches_per_second = (
                batch_count /
                train_elapsed
                if train_elapsed > 0
                else 0.0
            )

            (
                validation_loss,
                validation_mae,
                validation_samples,
                validation_elapsed,
            ) = validate(
                model,
                validation_loader,
                device,
                loss_function,
                args.max_validation_batches,
                use_amp,
            )

            elapsed = (
                time.perf_counter()
                -
                epoch_start
            )

            print(
                f"Train loss:       "
                f"{train_loss:.6f}"
            )

            print(
                f"Validation loss:  "
                f"{validation_loss:.6f}"
            )

            print(
                f"Validation MAE:   "
                f"{validation_mae:.2f} cp"
            )

            print(
                f"Train samples:    "
                f"{sample_count:,}"
            )

            print(
                f"Train throughput: "
                f"{train_samples_per_second:,.0f} "
                f"samples/sec"
            )

            print(
                f"Batch throughput: "
                f"{train_batches_per_second:,.2f} "
                f"batches/sec"
            )

            print(
                f"Validation time:  "
                f"{validation_elapsed:.1f} sec"
            )

            print(
                f"Epoch time:       "
                f"{elapsed:.1f} sec"
            )

            if device.type == "cuda":
                allocated = (
                    torch.cuda.memory_allocated()
                    /
                    (1024 ** 2)
                )

                reserved = (
                    torch.cuda.memory_reserved()
                    /
                    (1024 ** 2)
                )

                print(
                    f"CUDA memory:      "
                    f"{allocated:.1f} MiB allocated, "
                    f"{reserved:.1f} MiB reserved"
                )

            current_checkpoint = (
                output_dir /
                "checkpoint-last.pt"
            )

            save_checkpoint(
                current_checkpoint,
                model,
                optimizer,
                epoch,
                best_validation_loss,
            )

            # Export the latest network every epoch.
            quantized = quantize_model(
                model
            )

            # Quantization must never move the live training
            # model away from the selected training device.
            model_device = next(
                model.parameters()
            ).device

            device_mismatch = (
                model_device.type !=
                device.type
            )

            if (
                not device_mismatch
                and
                device.type == "cuda"
                and
                device.index is not None
            ):
                device_mismatch = (
                    model_device.index !=
                    device.index
                )

            if device_mismatch:
                raise RuntimeError(
                    "NNUE export changed the training "
                    "model device: "
                    f"expected {device}, "
                    f"got {model_device}"
                )

            save_quantized_network(
                quantized,
                output_dir /
                "knightbot-latest.nnue",
            )

            if (
                validation_loss <
                best_validation_loss
            ):
                best_validation_loss = (
                    validation_loss
                )

                best_checkpoint = (
                    output_dir /
                    "checkpoint-best.pt"
                )

                save_checkpoint(
                    best_checkpoint,
                    model,
                    optimizer,
                    epoch,
                    best_validation_loss,
                )

                save_quantized_network(
                    quantized,
                    output_dir /
                    "knightbot-best.nnue",
                )

                print(
                    "New best network."
                )

            metrics_writer.writerow(
                [
                    epoch,
                    train_loss,
                    validation_loss,
                    validation_mae,
                    elapsed,
                    sample_count,
                    train_samples_per_second,
                    train_batches_per_second,
                    validation_samples,
                    validation_elapsed,
                ]
            )

            metrics_file.flush()

    finally:
        metrics_file.close()


if __name__ == "__main__":
    main()