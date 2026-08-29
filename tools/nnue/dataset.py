from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import chess
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from knightbot_nnue import (
    INPUT_COUNT,
    active_features_from_board,
    stable_split_value,
)


PREENCODED_FORMAT_VERSION = 1

FEATURES_FILENAME = "features.bin"
OFFSETS_FILENAME = "offsets.bin"
TARGETS_FILENAME = "targets.bin"
TRAIN_INDICES_FILENAME = "train_indices.bin"
VALIDATION_INDICES_FILENAME = "validation_indices.bin"
METADATA_FILENAME = "meta.json"


class KnightBotDataset(IterableDataset):
    """
    Original TSV/FEN dataset.

    This remains available for debugging and small experiments.

    For large training runs, prefer PreencodedKnightBotDataset so that
    FEN parsing is not repeated every epoch.
    """

    def __init__(
        self,
        path: str | Path,
        split: str,
        validation_permyriad: int = 500,
        shuffle_buffer: int = 25000,
        seed: int = 2026,
        target_clamp: int = 2000,
    ) -> None:
        super().__init__()

        if split not in {
            "train",
            "validation",
        }:
            raise ValueError(
                "split must be train or validation"
            )

        if not (
            0 <
            validation_permyriad <
            10000
        ):
            raise ValueError(
                "validation_permyriad "
                "must be between 1 and 9999."
            )

        self.path = Path(path)
        self.split = split
        self.validation_permyriad = (
            validation_permyriad
        )
        self.shuffle_buffer = max(
            0,
            shuffle_buffer,
        )
        self.seed = seed
        self.target_clamp = target_clamp

    def _belongs_to_split(
        self,
        fen: str,
    ) -> bool:
        value = (
            stable_split_value(fen)
            %
            10000
        )

        is_validation = (
            value <
            self.validation_permyriad
        )

        if self.split == "validation":
            return is_validation

        return not is_validation

    def _read_samples(
        self,
    ) -> Iterator[
        tuple[list[int], float]
    ]:
        worker = torch.utils.data.get_worker_info()

        if worker is None:
            worker_id = 0
            worker_count = 1
        else:
            worker_id = worker.id
            worker_count = worker.num_workers

        with self.path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as source:
            for line_index, line in enumerate(source):
                # Properly shard IterableDataset work across DataLoader
                # workers. Without this, every worker would read the
                # complete file and duplicate samples.
                if (
                    line_index %
                    worker_count
                    !=
                    worker_id
                ):
                    continue

                line = line.rstrip(
                    "\r\n"
                )

                if not line:
                    continue

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

                    if not self._belongs_to_split(
                        fen
                    ):
                        continue

                    board = chess.Board(
                        fen
                    )

                except (
                    ValueError,
                    IndexError,
                ):
                    continue

                target = max(
                    -self.target_clamp,
                    min(
                        self.target_clamp,
                        target,
                    ),
                )

                yield (
                    active_features_from_board(
                        board
                    ),
                    float(target),
                )

    def __iter__(
        self,
    ) -> Iterator[
        tuple[list[int], float]
    ]:
        stream = self._read_samples()

        if (
            self.split != "train"
            or
            self.shuffle_buffer <= 1
        ):
            yield from stream
            return

        worker = torch.utils.data.get_worker_info()

        worker_id = (
            0
            if worker is None
            else worker.id
        )

        rng = random.Random(
            self.seed +
            worker_id
        )

        buffer: list[
            tuple[list[int], float]
        ] = []

        for sample in stream:
            if (
                len(buffer) <
                self.shuffle_buffer
            ):
                buffer.append(
                    sample
                )
                continue

            index = rng.randrange(
                len(buffer)
            )

            yield buffer[index]

            buffer[index] = sample

        rng.shuffle(
            buffer
        )

        yield from buffer


class PreencodedStore:
    """
    Low-level memory-mapped access to a preencoded KnightBot dataset.

    The files remain on disk and are mapped into the process address
    space. The entire 5M-position dataset is not copied into RAM.
    """

    def __init__(
        self,
        directory: str | Path,
    ) -> None:
        self.directory = Path(directory)

        self.metadata_path = (
            self.directory /
            METADATA_FILENAME
        )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Missing metadata file: "
                f"{self.metadata_path}"
            )

        with self.metadata_path.open(
            "r",
            encoding="utf-8",
        ) as source:
            self.metadata = json.load(
                source
            )

        version = int(
            self.metadata.get(
                "format_version",
                -1,
            )
        )

        if version != PREENCODED_FORMAT_VERSION:
            raise ValueError(
                "Unsupported preencoded dataset "
                f"version: {version}"
            )

        input_count = int(
            self.metadata.get(
                "input_count",
                -1,
            )
        )

        if input_count != INPUT_COUNT:
            raise ValueError(
                "Preencoded input feature count "
                f"mismatch: {input_count} "
                f"!= {INPUT_COUNT}"
            )

        self.sample_count = int(
            self.metadata[
                "sample_count"
            ]
        )

        self.feature_count = int(
            self.metadata[
                "feature_count"
            ]
        )

        self.validation_permyriad = int(
            self.metadata[
                "validation_permyriad"
            ]
        )

        self.train_count = int(
            self.metadata[
                "train_count"
            ]
        )

        self.validation_count = int(
            self.metadata[
                "validation_count"
            ]
        )

        self.features_path = (
            self.directory /
            FEATURES_FILENAME
        )

        self.offsets_path = (
            self.directory /
            OFFSETS_FILENAME
        )

        self.targets_path = (
            self.directory /
            TARGETS_FILENAME
        )

        self.train_indices_path = (
            self.directory /
            TRAIN_INDICES_FILENAME
        )

        self.validation_indices_path = (
            self.directory /
            VALIDATION_INDICES_FILENAME
        )

        required_paths = [
            self.features_path,
            self.offsets_path,
            self.targets_path,
            self.train_indices_path,
            self.validation_indices_path,
        ]

        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing preencoded file: {path}"
                )

        expected_feature_bytes = (
            self.feature_count *
            np.dtype("<u2").itemsize
        )

        expected_offset_bytes = (
            (self.sample_count + 1) *
            np.dtype("<u8").itemsize
        )

        expected_target_bytes = (
            self.sample_count *
            np.dtype("<i2").itemsize
        )

        expected_train_index_bytes = (
            self.train_count *
            np.dtype("<u4").itemsize
        )

        expected_validation_index_bytes = (
            self.validation_count *
            np.dtype("<u4").itemsize
        )

        checks = [
            (
                self.features_path,
                expected_feature_bytes,
            ),
            (
                self.offsets_path,
                expected_offset_bytes,
            ),
            (
                self.targets_path,
                expected_target_bytes,
            ),
            (
                self.train_indices_path,
                expected_train_index_bytes,
            ),
            (
                self.validation_indices_path,
                expected_validation_index_bytes,
            ),
        ]

        for path, expected_size in checks:
            actual_size = path.stat().st_size

            if actual_size != expected_size:
                raise ValueError(
                    f"Unexpected size for {path.name}: "
                    f"{actual_size} bytes, expected "
                    f"{expected_size} bytes"
                )

        self._features: np.memmap | None = None
        self._offsets: np.memmap | None = None
        self._targets: np.memmap | None = None

    def _ensure_open(self) -> None:
        if self._features is None:
            self._features = np.memmap(
                self.features_path,
                dtype="<u2",
                mode="r",
            )

        if self._offsets is None:
            self._offsets = np.memmap(
                self.offsets_path,
                dtype="<u8",
                mode="r",
            )

        if self._targets is None:
            self._targets = np.memmap(
                self.targets_path,
                dtype="<i2",
                mode="r",
            )

    def features_for_sample(
        self,
        sample_index: int,
    ) -> np.ndarray:
        if not (
            0 <=
            sample_index <
            self.sample_count
        ):
            raise IndexError(
                sample_index
            )

        self._ensure_open()

        assert self._features is not None
        assert self._offsets is not None

        start = int(
            self._offsets[
                sample_index
            ]
        )

        end = int(
            self._offsets[
                sample_index + 1
            ]
        )

        return np.asarray(
            self._features[
                start:end
            ],
            dtype=np.uint16,
        )

    def target_for_sample(
        self,
        sample_index: int,
    ) -> int:
        if not (
            0 <=
            sample_index <
            self.sample_count
        ):
            raise IndexError(
                sample_index
            )

        self._ensure_open()

        assert self._targets is not None

        return int(
            self._targets[
                sample_index
            ]
        )

    def __getstate__(self):
        state = self.__dict__.copy()

        # Reopen memory maps separately inside DataLoader worker
        # processes rather than trying to pickle active mappings.
        state["_features"] = None
        state["_offsets"] = None
        state["_targets"] = None

        return state


class PreencodedKnightBotDataset(Dataset):
    """
    Map-style training/validation dataset backed by memory-mapped files.

    This is the preferred dataset for the 5M-position training run.
    """

    def __init__(
        self,
        directory: str | Path,
        split: str,
        target_clamp: int = 2000,
    ) -> None:
        super().__init__()

        if split not in {
            "train",
            "validation",
        }:
            raise ValueError(
                "split must be train or validation"
            )

        self.directory = Path(directory)
        self.split = split
        self.target_clamp = int(
            target_clamp
        )

        self.store = PreencodedStore(
            self.directory
        )

        if split == "train":
            self.indices_path = (
                self.store.train_indices_path
            )
            self._length = (
                self.store.train_count
            )
        else:
            self.indices_path = (
                self.store.validation_indices_path
            )
            self._length = (
                self.store.validation_count
            )

        self._indices: np.memmap | None = None

    def _ensure_indices_open(
        self,
    ) -> None:
        if self._indices is None:
            self._indices = np.memmap(
                self.indices_path,
                dtype="<u4",
                mode="r",
            )

    def __len__(self) -> int:
        return self._length

    def __getitem__(
        self,
        index: int,
    ) -> tuple[np.ndarray, float]:
        if index < 0:
            index += self._length

        if not (
            0 <=
            index <
            self._length
        ):
            raise IndexError(
                index
            )

        self._ensure_indices_open()

        assert self._indices is not None

        sample_index = int(
            self._indices[
                index
            ]
        )

        features = (
            self.store.features_for_sample(
                sample_index
            )
        )

        target = (
            self.store.target_for_sample(
                sample_index
            )
        )

        target = max(
            -self.target_clamp,
            min(
                self.target_clamp,
                target,
            ),
        )

        return (
            features,
            float(target),
        )

    def __getstate__(self):
        state = self.__dict__.copy()

        state["_indices"] = None

        return state


def collate_sparse(
    samples: list[
        tuple[
            list[int] | np.ndarray,
            float,
        ]
    ],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Convert variable-length sparse feature lists into EmbeddingBag input.

    No dense [batch, 768] matrix is created.
    """

    batch_size = len(samples)

    if batch_size == 0:
        return (
            torch.empty(
                0,
                dtype=torch.long,
            ),
            torch.empty(
                0,
                dtype=torch.long,
            ),
            torch.empty(
                0,
                dtype=torch.float32,
            ),
        )

    lengths = np.fromiter(
        (
            len(features)
            for features, _ in samples
        ),
        dtype=np.int64,
        count=batch_size,
    )

    offsets = np.empty(
        batch_size,
        dtype=np.int64,
    )

    offsets[0] = 0

    if batch_size > 1:
        np.cumsum(
            lengths[:-1],
            out=offsets[1:],
        )

    total_features = int(
        lengths.sum()
    )

    flat_features = np.empty(
        total_features,
        dtype=np.int64,
    )

    targets = np.empty(
        batch_size,
        dtype=np.float32,
    )

    cursor = 0

    for sample_index, (
        features,
        target,
    ) in enumerate(samples):
        feature_count = len(
            features
        )

        if feature_count:
            flat_features[
                cursor:
                cursor + feature_count
            ] = np.asarray(
                features,
                dtype=np.int64,
            )

        cursor += feature_count

        targets[
            sample_index
        ] = float(
            target
        )

    return (
        torch.from_numpy(
            flat_features
        ),
        torch.from_numpy(
            offsets
        ),
        torch.from_numpy(
            targets
        ),
    )