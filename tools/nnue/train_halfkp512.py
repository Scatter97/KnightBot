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

from dataset_halfkp512 import (
    HalfKP512Dataset,
    HalfKP512Store,
    collate_halfkp512,
)

from halfkp_512 import (
    HalfKP512,
    quantize_model,
    save_quantized_network,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train KnightBot HalfKP-512."
        )
    )

    parser.add_argument(
        "--preencoded-dir",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default="runs/halfkp-512",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
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
        "--target-clamp",
        type=int,
        default=3000,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--resume",
        default="",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    parser.add_argument(
        "--benchmark-loader",
        action="store_true",
    )

    parser.add_argument(
        "--benchmark-batches",
        type=int,
        default=100,
    )

    return parser.parse_args()


def make_loader(
    dataset,
    args,
    shuffle: bool,
):
    kwargs = {
        "dataset": dataset,
        "batch_size": (
            args.batch_size
        ),
        "shuffle": shuffle,
        "collate_fn": (
            collate_halfkp512
        ),
        "num_workers": (
            args.num_workers
        ),
        "pin_memory": True,
    }

    if args.num_workers > 0:
        kwargs[
            "prefetch_factor"
        ] = args.prefetch_factor

        kwargs[
            "persistent_workers"
        ] = True

    return DataLoader(
        **kwargs
    )


def move_batch(
    batch,
    device,
):
    (
        white_features,
        black_features,
        offsets,
        context,
        targets,
    ) = batch

    return (
        white_features.to(
            device,
            non_blocking=True,
        ),
        black_features.to(
            device,
            non_blocking=True,
        ),
        offsets.to(
            device,
            non_blocking=True,
        ),
        context.to(
            device,
            non_blocking=True,
        ),
        targets.to(
            device,
            non_blocking=True,
        ),
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_validation_loss,
):
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


@torch.no_grad()
def validate(
    model,
    loader,
    device,
    loss_function,
    use_amp,
    max_batches,
):
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
            white_features,
            black_features,
            offsets,
            context,
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
                white_features,
                black_features,
                offsets,
                context,
            )

            loss = loss_function(
                predictions,
                targets,
            )

        batch_size = int(
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


def benchmark_loader(
    loader,
    max_batches,
):
    start = time.perf_counter()

    samples = 0
    batches = 0

    for batch_index, batch in enumerate(
        tqdm(
            loader,
            desc="Loader benchmark",
        )
    ):
        if (
            max_batches > 0
            and
            batch_index >= max_batches
        ):
            break

        samples += int(
            batch[-1].numel()
        )

        batches += 1

    elapsed = (
        time.perf_counter() -
        start
    )

    print()
    print(
        f"Batches:      {batches:,}"
    )
    print(
        f"Samples:      {samples:,}"
    )
    print(
        f"Elapsed:      {elapsed:.3f} sec"
    )
    print(
        f"Samples/sec:  "
        f"{samples / elapsed:,.0f}"
    )


def main() -> None:
    args = parse_args()

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
            "WARNING: CUDA not found."
        )

    use_amp = (
        device.type == "cuda"
        and
        not args.no_amp
    )

    store = HalfKP512Store(
        args.preencoded_dir
    )

    train_dataset = HalfKP512Dataset(
        args.preencoded_dir,
        split="train",
        target_clamp=(
            args.target_clamp
        ),
    )

    validation_dataset = HalfKP512Dataset(
        args.preencoded_dir,
        split="validation",
        target_clamp=(
            args.target_clamp
        ),
    )

    print(
        "Architecture:      HalfKP-512"
    )
    print(
        f"Total positions:   "
        f"{store.sample_count:,}"
    )
    print(
        f"Train positions:   "
        f"{store.train_count:,}"
    )
    print(
        f"Validation:        "
        f"{store.validation_count:,}"
    )
    print(
        f"Workers:           "
        f"{args.num_workers}"
    )
    print(
        f"AMP:               "
        f"{'on' if use_amp else 'off'}"
    )

    train_loader = make_loader(
        train_dataset,
        args,
        shuffle=True,
    )

    validation_loader = make_loader(
        validation_dataset,
        args,
        shuffle=False,
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

    model = HalfKP512().to(
        device
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Parameters:        "
        f"{parameter_count:,}"
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

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
        for param_group in optimizer.param_groups:
            param_group["lr"] = (
                args.learning_rate
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
        bool(args.resume)
        and
        metrics_path.exists()
    )

    metrics_file = metrics_path.open(
        "a"
        if metrics_exists
        else
        "w",
        encoding="utf-8",
        newline="",
    )

    writer = csv.writer(
        metrics_file
    )

    if not metrics_exists:
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "validation_loss",
                "validation_mae_cp",
                "epoch_seconds",
                "train_samples_per_sec",
            ]
        )

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

            start = time.perf_counter()

            running_loss = 0.0
            sample_count = 0

            progress = tqdm(
                train_loader,
                desc="Training",
            )

            for batch_index, batch in enumerate(
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
                    white_features,
                    black_features,
                    offsets,
                    context,
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
                        white_features,
                        black_features,
                        offsets,
                        context,
                    )

                    loss = loss_function(
                        predictions,
                        targets,
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

                batch_size = int(
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

                elapsed = (
                    time.perf_counter() -
                    start
                )

                progress.set_postfix(
                    loss=(
                        f"{running_loss / sample_count:.4f}"
                    ),
                    samples_s=(
                        f"{sample_count / elapsed:,.0f}"
                    ),
                )

            train_elapsed = (
                time.perf_counter() -
                start
            )

            train_loss = (
                running_loss /
                sample_count
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
                use_amp,
                args.max_validation_batches,
            )

            epoch_elapsed = (
                time.perf_counter() -
                start
            )

            throughput = (
                sample_count /
                train_elapsed
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
                f"{throughput:,.0f} samples/sec"
            )
            print(
                f"Validation time:  "
                f"{validation_elapsed:.1f} sec"
            )
            print(
                f"Epoch time:       "
                f"{epoch_elapsed:.1f} sec"
            )

            if device.type == "cuda":
                print(
                    f"CUDA memory:      "
                    f"{torch.cuda.memory_allocated() / (1024 ** 2):.1f} "
                    f"MiB allocated, "
                    f"{torch.cuda.memory_reserved() / (1024 ** 2):.1f} "
                    f"MiB reserved"
                )

                is_best = (
                validation_loss <
                best_validation_loss
            )

            if is_best:
                best_validation_loss = (
                    validation_loss
                )

            last_checkpoint = (
                output_dir /
                "checkpoint-last.pt"
            )

            save_checkpoint(
                last_checkpoint,
                model,
                optimizer,
                epoch,
                best_validation_loss,
            )

            quantized = quantize_model(
                model
            )

            model_device = next(
                model.parameters()
            ).device

            if (
                model_device.type !=
                device.type
            ):
                raise RuntimeError(
                    "Quantization changed model device."
                )

            save_quantized_network(
                quantized,
                output_dir /
                "halfkp-512-latest.nnue",
            )

            if is_best:
                save_checkpoint(
                    output_dir /
                    "checkpoint-best.pt",
                    model,
                    optimizer,
                    epoch,
                    best_validation_loss,
                )

                save_quantized_network(
                    quantized,
                    output_dir /
                    "halfkp-512-best.nnue",
                )

                print(
                    "New best network."
                )

            writer.writerow(
                [
                    epoch,
                    train_loss,
                    validation_loss,
                    validation_mae,
                    epoch_elapsed,
                    throughput,
                ]
            )

            metrics_file.flush()

    finally:
        metrics_file.close()


if __name__ == "__main__":
    main()