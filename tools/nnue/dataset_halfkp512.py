from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from halfkp_512 import (
    CONTEXT_COUNT,
    HALFKP_FEATURE_COUNT,
)


FORMAT_VERSION = 2

WHITE_FEATURES_FILENAME = "white_features.bin"
BLACK_FEATURES_FILENAME = "black_features.bin"
OFFSETS_FILENAME = "offsets.bin"
CONTEXT_FILENAME = "context.bin"
TARGETS_FILENAME = "targets.bin"
TRAIN_INDICES_FILENAME = "train_indices.bin"
VALIDATION_INDICES_FILENAME = "validation_indices.bin"
METADATA_FILENAME = "meta.json"


class HalfKP512Store:
    def __init__(
        self,
        directory: str | Path,
    ) -> None:
        self.directory = Path(
            directory
        )

        metadata_path = (
            self.directory /
            METADATA_FILENAME
        )

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as source:
            self.metadata = json.load(
                source
            )

        if int(
            self.metadata[
                "format_version"
            ]
        ) != FORMAT_VERSION:
            raise ValueError(
                "Unsupported HalfKP dataset format."
            )

        if int(
            self.metadata[
                "feature_count"
            ]
        ) != HALFKP_FEATURE_COUNT:
            raise ValueError(
                "HalfKP feature count mismatch."
            )

        self.sample_count = int(
            self.metadata[
                "sample_count"
            ]
        )

        self.flat_feature_count = int(
            self.metadata[
                "flat_feature_count"
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

        self.validation_permyriad = int(
            self.metadata[
                "validation_permyriad"
            ]
        )

        self.white_features_path = (
            self.directory /
            WHITE_FEATURES_FILENAME
        )

        self.black_features_path = (
            self.directory /
            BLACK_FEATURES_FILENAME
        )

        self.offsets_path = (
            self.directory /
            OFFSETS_FILENAME
        )

        self.context_path = (
            self.directory /
            CONTEXT_FILENAME
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

        self._white_features = None
        self._black_features = None
        self._offsets = None
        self._context = None
        self._targets = None

    def _ensure_open(self) -> None:
        if self._white_features is None:
            self._white_features = np.memmap(
                self.white_features_path,
                dtype="<u2",
                mode="r",
            )

        if self._black_features is None:
            self._black_features = np.memmap(
                self.black_features_path,
                dtype="<u2",
                mode="r",
            )

        if self._offsets is None:
            self._offsets = np.memmap(
                self.offsets_path,
                dtype="<u8",
                mode="r",
            )

        if self._context is None:
            self._context = np.memmap(
                self.context_path,
                dtype="<u2",
                mode="r",
            )

        if self._targets is None:
            self._targets = np.memmap(
                self.targets_path,
                dtype="<i2",
                mode="r",
            )

    def sample(
        self,
        sample_index: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        int,
        int,
    ]:
        self._ensure_open()

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

        return (
            np.asarray(
                self._white_features[
                    start:end
                ],
                dtype=np.uint16,
            ),
            np.asarray(
                self._black_features[
                    start:end
                ],
                dtype=np.uint16,
            ),
            int(
                self._context[
                    sample_index
                ]
            ),
            int(
                self._targets[
                    sample_index
                ]
            ),
        )

    def __getstate__(self):
        state = self.__dict__.copy()

        state["_white_features"] = None
        state["_black_features"] = None
        state["_offsets"] = None
        state["_context"] = None
        state["_targets"] = None

        return state


class HalfKP512Dataset(Dataset):
    def __init__(
        self,
        directory: str | Path,
        split: str,
        target_clamp: int = 3000,
    ) -> None:
        super().__init__()

        if split not in {
            "train",
            "validation",
        }:
            raise ValueError(
                "split must be train or validation"
            )

        self.store = HalfKP512Store(
            directory
        )

        self.split = split
        self.target_clamp = int(
            target_clamp
        )

        if split == "train":
            self.indices_path = (
                self.store.train_indices_path
            )

            self.length = (
                self.store.train_count
            )

        else:
            self.indices_path = (
                self.store.validation_indices_path
            )

            self.length = (
                self.store.validation_count
            )

        self._indices = None

    def _ensure_indices(self) -> None:
        if self._indices is None:
            self._indices = np.memmap(
                self.indices_path,
                dtype="<u4",
                mode="r",
            )

    def __len__(self) -> int:
        return self.length

    def __getitem__(
        self,
        index: int,
    ):
        self._ensure_indices()

        sample_index = int(
            self._indices[
                index
            ]
        )

        (
            white_features,
            black_features,
            context_bits,
            target,
        ) = self.store.sample(
            sample_index
        )

        target = max(
            -self.target_clamp,
            min(
                self.target_clamp,
                target,
            ),
        )

        return (
            white_features,
            black_features,
            context_bits,
            float(target),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_indices"] = None
        return state


def collate_halfkp512(
    samples,
):
    batch_size = len(
        samples
    )

    lengths = np.fromiter(
        (
            len(sample[0])
            for sample in samples
        ),
        dtype=np.int64,
        count=batch_size,
    )

    offsets = np.zeros(
        batch_size,
        dtype=np.int64,
    )

    if batch_size > 1:
        np.cumsum(
            lengths[:-1],
            out=offsets[1:],
        )

    total_features = int(
        lengths.sum()
    )

    white_flat = np.empty(
        total_features,
        dtype=np.int64,
    )

    black_flat = np.empty(
        total_features,
        dtype=np.int64,
    )

    context = np.zeros(
        (
            batch_size,
            CONTEXT_COUNT,
        ),
        dtype=np.float32,
    )

    targets = np.empty(
        batch_size,
        dtype=np.float32,
    )

    cursor = 0

    for batch_index, (
        white_features,
        black_features,
        context_bits,
        target,
    ) in enumerate(samples):
        count = len(
            white_features
        )

        if len(
            black_features
        ) != count:
            raise RuntimeError(
                "White/black HalfKP feature count mismatch."
            )

        white_flat[
            cursor:
            cursor + count
        ] = white_features

        black_flat[
            cursor:
            cursor + count
        ] = black_features

        cursor += count

        for context_index in range(
            CONTEXT_COUNT
        ):
            if (
                context_bits &
                (1 << context_index)
            ):
                context[
                    batch_index,
                    context_index,
                ] = 1.0

        targets[
            batch_index
        ] = target

    return (
        torch.from_numpy(
            white_flat
        ),
        torch.from_numpy(
            black_flat
        ),
        torch.from_numpy(
            offsets
        ),
        torch.from_numpy(
            context
        ),
        torch.from_numpy(
            targets
        ),
    )