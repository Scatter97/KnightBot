from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chess
import numpy as np
import torch
from torch import nn


NNUE_MAGIC = b"KNUE0001"
NNUE_FORMAT_VERSION = 1

INPUT_COUNT = 12 * 64
HIDDEN_COUNT = 128

ACTIVATION_MAX = 127

# Quantization scales used when exporting the trained float model.
#
# Hidden float values are multiplied by HIDDEN_SCALE.
# Output weights are multiplied by OUTPUT_WEIGHT_SCALE.
#
# C++ then computes:
#
#   cp =
#       (
#           output_bias_int
#           +
#           sum(hidden_int * output_weight_int)
#       )
#       /
#       OUTPUT_SCALE
#
HIDDEN_SCALE = 32
OUTPUT_WEIGHT_SCALE = 64
OUTPUT_SCALE = HIDDEN_SCALE * OUTPUT_WEIGHT_SCALE


PIECE_TO_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def piece_index(piece: chess.Piece) -> int:
    base = PIECE_TO_INDEX[piece.piece_type]

    if piece.color == chess.WHITE:
        return base

    return 6 + base


def feature_index(piece: chess.Piece, square: chess.Square) -> int:
    return piece_index(piece) * 64 + int(square)


def active_features_from_board(board: chess.Board) -> list[int]:
    features: list[int] = []

    for square, piece in board.piece_map().items():
        features.append(
            feature_index(piece, square)
        )

    return features


def active_features_from_fen(fen: str) -> list[int]:
    board = chess.Board(fen)

    return active_features_from_board(board)


def normalized_position_key(board: chess.Board) -> str:
    """
    Position identity used for training-data deduplication.

    Keeps:
        board
        side to move
        castling rights
        en-passant square

    Drops:
        halfmove clock
        fullmove number

    Those clocks do not change the static chess position represented
    by the NNUE feature encoder.
    """

    fields = board.fen(
        en_passant="fen"
    ).split()

    return " ".join(fields[:4])


def stable_split_value(text: str) -> int:
    digest = hashlib.blake2b(
        text.encode("utf-8"),
        digest_size=8,
    ).digest()

    return int.from_bytes(
        digest,
        byteorder="little",
        signed=False,
    )


class KnightBotNNUE(nn.Module):
    """
    Exact trainable equivalent of:

        768 binary piece-square features
                    |
                    v
             128 hidden units
                    |
              clipped ReLU
                    |
                    v
               scalar CP

    EmbeddingBag is used instead of constructing a dense 768-element
    one-hot vector for every chess position.

    Summing embeddings for the active features is mathematically the
    same as multiplying the binary feature vector by the first layer.
    """

    def __init__(self) -> None:
        super().__init__()

        self.feature_weights = nn.EmbeddingBag(
            INPUT_COUNT,
            HIDDEN_COUNT,
            mode="sum",
            include_last_offset=False,
        )

        self.hidden_bias = nn.Parameter(
            torch.zeros(
                HIDDEN_COUNT,
                dtype=torch.float32,
            )
        )

        self.output = nn.Linear(
            HIDDEN_COUNT,
            1,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.feature_weights.weight,
            mean=0.0,
            std=0.02,
        )

        nn.init.zeros_(
            self.hidden_bias
        )

        nn.init.normal_(
            self.output.weight,
            mean=0.0,
            std=0.02,
        )

        nn.init.zeros_(
            self.output.bias
        )

    def forward(
        self,
        feature_indices: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.feature_weights(
            feature_indices,
            offsets,
        )

        hidden = hidden + self.hidden_bias

        # Float training representation corresponding to the C++
        # integer range 0..127 after multiplication by HIDDEN_SCALE.
        hidden_float_max = (
            ACTIVATION_MAX /
            HIDDEN_SCALE
        )

        hidden = torch.clamp(
            hidden,
            min=0.0,
            max=hidden_float_max,
        )

        output = self.output(hidden)

        return output.squeeze(-1)


@dataclass
class QuantizedNetwork:
    hidden_bias: np.ndarray
    input_weights: np.ndarray
    output_weights: np.ndarray
    output_bias: int
    activation_max: int
    output_scale: int


def _clip_int16(values: np.ndarray) -> np.ndarray:
    return np.clip(
        values,
        np.iinfo(np.int16).min,
        np.iinfo(np.int16).max,
    ).astype(np.int16)


def _clip_int32_scalar(value: int) -> int:
    return int(
        np.clip(
            value,
            np.iinfo(np.int32).min,
            np.iinfo(np.int32).max,
        )
    )


def quantize_model(
    model: KnightBotNNUE,
) -> QuantizedNetwork:
    """
    Quantize a trained float model without changing the device
    or training/eval state of the live model.

    This function is called during training after every epoch.
    It must therefore NEVER call model.cpu(), model.cuda(),
    model.train(), or model.eval() on the original model.
    """

    with torch.no_grad():
        feature_weights = (
            model.feature_weights.weight
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        hidden_bias = (
            model.hidden_bias
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        output_weights = (
            model.output.weight[0]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        output_bias = float(
            model.output.bias[0]
            .detach()
            .cpu()
            .item()
        )

    q_input = np.rint(
        feature_weights *
        HIDDEN_SCALE
    )

    q_hidden_bias = np.rint(
        hidden_bias *
        HIDDEN_SCALE
    )

    q_output = np.rint(
        output_weights *
        OUTPUT_WEIGHT_SCALE
    )

    q_output_bias = round(
        output_bias *
        OUTPUT_SCALE
    )

    q_input = _clip_int16(
        q_input
    )

    q_hidden_bias = np.clip(
        q_hidden_bias,
        np.iinfo(np.int32).min,
        np.iinfo(np.int32).max,
    ).astype(np.int32)

    q_output = _clip_int16(
        q_output
    )

    q_output_bias = _clip_int32_scalar(
        q_output_bias
    )

    return QuantizedNetwork(
        hidden_bias=q_hidden_bias,
        input_weights=q_input,
        output_weights=q_output,
        output_bias=q_output_bias,
        activation_max=ACTIVATION_MAX,
        output_scale=OUTPUT_SCALE,
    )


def save_quantized_network(
    network: QuantizedNetwork,
    path: str | Path,
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if network.input_weights.shape != (
        INPUT_COUNT,
        HIDDEN_COUNT,
    ):
        raise ValueError(
            "Unexpected input weight shape: "
            f"{network.input_weights.shape}"
        )

    if network.hidden_bias.shape != (
        HIDDEN_COUNT,
    ):
        raise ValueError(
            "Unexpected hidden bias shape."
        )

    if network.output_weights.shape != (
        HIDDEN_COUNT,
    ):
        raise ValueError(
            "Unexpected output weight shape."
        )

    with path.open("wb") as output:
        output.write(
            NNUE_MAGIC
        )

        output.write(
            struct.pack(
                "<IIIii",
                NNUE_FORMAT_VERSION,
                INPUT_COUNT,
                HIDDEN_COUNT,
                network.activation_max,
                network.output_scale,
            )
        )

        output.write(
            network.hidden_bias
            .astype("<i4")
            .tobytes(order="C")
        )

        # The C++ loader expects feature-major weights:
        #
        #     feature * hidden_count + hidden
        output.write(
            network.input_weights
            .astype("<i2")
            .tobytes(order="C")
        )

        output.write(
            network.output_weights
            .astype("<i2")
            .tobytes(order="C")
        )

        output.write(
            struct.pack(
                "<i",
                network.output_bias,
            )
        )


def load_quantized_network(
    path: str | Path,
) -> QuantizedNetwork:
    path = Path(path)

    with path.open("rb") as source:
        magic = source.read(8)

        if magic != NNUE_MAGIC:
            raise ValueError(
                f"Bad NNUE magic: {magic!r}"
            )

        header = source.read(
            struct.calcsize("<IIIii")
        )

        (
            version,
            input_count,
            hidden_count,
            activation_max,
            output_scale,
        ) = struct.unpack(
            "<IIIii",
            header,
        )

        if version != NNUE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported version: {version}"
            )

        if input_count != INPUT_COUNT:
            raise ValueError(
                f"Input count mismatch: {input_count}"
            )

        if hidden_count != HIDDEN_COUNT:
            raise ValueError(
                f"Hidden count mismatch: {hidden_count}"
            )

        hidden_bias = np.frombuffer(
            source.read(
                HIDDEN_COUNT * 4
            ),
            dtype="<i4",
        ).copy()

        input_weights = np.frombuffer(
            source.read(
                INPUT_COUNT *
                HIDDEN_COUNT *
                2
            ),
            dtype="<i2",
        ).copy()

        input_weights = input_weights.reshape(
            INPUT_COUNT,
            HIDDEN_COUNT,
        )

        output_weights = np.frombuffer(
            source.read(
                HIDDEN_COUNT * 2
            ),
            dtype="<i2",
        ).copy()

        output_bias_raw = source.read(4)

        if len(output_bias_raw) != 4:
            raise ValueError(
                "NNUE file ended before output bias."
            )

        output_bias = struct.unpack(
            "<i",
            output_bias_raw,
        )[0]

        trailing = source.read(1)

        if trailing:
            raise ValueError(
                "Unexpected trailing data in NNUE file."
            )

    return QuantizedNetwork(
        hidden_bias=hidden_bias,
        input_weights=input_weights,
        output_weights=output_weights,
        output_bias=output_bias,
        activation_max=activation_max,
        output_scale=output_scale,
    )


def evaluate_quantized_features(
    network: QuantizedNetwork,
    features: Iterable[int],
) -> int:
    accumulator = (
        network.hidden_bias
        .astype(np.int64)
        .copy()
    )

    for feature in features:
        if not 0 <= feature < INPUT_COUNT:
            raise ValueError(
                f"Feature out of range: {feature}"
            )

        accumulator += (
            network.input_weights[
                feature
            ].astype(np.int64)
        )

    accumulator = np.clip(
        accumulator,
        0,
        network.activation_max,
    )

    output = int(
        network.output_bias
    )

    output += int(
        np.sum(
            accumulator *
            network.output_weights.astype(
                np.int64
            )
        )
    )

    # C++ integer division truncates toward zero.
    if output >= 0:
        cp = (
            output //
            network.output_scale
        )
    else:
        cp = -(
            (-output) //
            network.output_scale
        )

    return max(
        -30000,
        min(
            30000,
            int(cp),
        ),
    )


def evaluate_quantized_fen(
    network: QuantizedNetwork,
    fen: str,
) -> int:
    return evaluate_quantized_features(
        network,
        active_features_from_fen(fen),
    )
