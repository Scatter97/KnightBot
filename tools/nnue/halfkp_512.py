from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chess
import numpy as np
import torch
from torch import nn


# ============================================================
# NETWORK FORMAT
# ============================================================

NNUE_MAGIC = b"KNUE0002"
NNUE_FORMAT_VERSION = 2

HALFKP_FEATURE_COUNT = 64 * 10 * 64
TRANSFORMER_HIDDEN = 512
CONTEXT_COUNT = 13
DENSE1_COUNT = 64
DENSE2_COUNT = 32

ACTIVATION_SCALE = 1024

ACTIVATION_MAX = (
    127 * ACTIVATION_SCALE // 32
)

WEIGHT_SCALE = 256

OUTPUT_SCALE = (
    ACTIVATION_SCALE *
    WEIGHT_SCALE
)

CONTEXT_ACTIVE = (
    ACTIVATION_SCALE
)

PIECE_TYPE_TO_RELATIVE_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}


# ============================================================
# FEATURE ENCODING
# ============================================================

def orient_square(
    perspective: chess.Color,
    square: chess.Square,
) -> int:
    """
    White perspective keeps the board unchanged.

    Black perspective vertically mirrors the board so equivalent
    king/piece structures share the same feature-transformer weights.
    """
    square = int(square)

    if perspective == chess.WHITE:
        return square

    return square ^ 56


def halfkp_piece_class(
    perspective: chess.Color,
    piece: chess.Piece,
) -> int:
    if piece.piece_type == chess.KING:
        raise ValueError(
            "Kings are not ordinary HalfKP piece features."
        )

    base = PIECE_TYPE_TO_RELATIVE_INDEX[
        piece.piece_type
    ]

    if piece.color == perspective:
        return base

    return 5 + base


def halfkp_feature_index(
    perspective: chess.Color,
    king_square: chess.Square,
    piece: chess.Piece,
    piece_square: chess.Square,
) -> int:
    oriented_king = orient_square(
        perspective,
        king_square,
    )

    oriented_piece = orient_square(
        perspective,
        piece_square,
    )

    piece_class = halfkp_piece_class(
        perspective,
        piece,
    )

    feature = (
        oriented_king * (10 * 64)
        +
        piece_class * 64
        +
        oriented_piece
    )

    if not (
        0 <=
        feature <
        HALFKP_FEATURE_COUNT
    ):
        raise ValueError(
            f"HalfKP feature out of range: {feature}"
        )

    return feature


def halfkp_features_from_board(
    board: chess.Board,
    perspective: chess.Color,
) -> list[int]:
    king_square = board.king(
        perspective
    )

    if king_square is None:
        raise ValueError(
            "Position has no king for perspective."
        )

    features: list[int] = []

    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue

        features.append(
            halfkp_feature_index(
                perspective,
                king_square,
                piece,
                square,
            )
        )

    # python-chess piece_map order is not part of our file format.
    # Sorting gives deterministic Python/C++ verification.
    features.sort()

    return features


def halfkp_features_from_fen(
    fen: str,
) -> tuple[
    list[int],
    list[int],
]:
    board = chess.Board(
        fen
    )

    return (
        halfkp_features_from_board(
            board,
            chess.WHITE,
        ),
        halfkp_features_from_board(
            board,
            chess.BLACK,
        ),
    )


# ============================================================
# CONTEXT FEATURES
# ============================================================

def context_bits_from_board(
    board: chess.Board,
) -> int:
    """
    Bits:

      0  White to move
      1  White kingside castling
      2  White queenside castling
      3  Black kingside castling
      4  Black queenside castling
      5  en-passant file a
      ...
      12 en-passant file h
    """

    bits = 0

    if board.turn == chess.WHITE:
        bits |= 1 << 0

    if board.has_kingside_castling_rights(
        chess.WHITE
    ):
        bits |= 1 << 1

    if board.has_queenside_castling_rights(
        chess.WHITE
    ):
        bits |= 1 << 2

    if board.has_kingside_castling_rights(
        chess.BLACK
    ):
        bits |= 1 << 3

    if board.has_queenside_castling_rights(
        chess.BLACK
    ):
        bits |= 1 << 4

    if board.ep_square is not None:
        ep_file = chess.square_file(
            board.ep_square
        )

        bits |= (
            1 <<
            (5 + ep_file)
        )

    return bits


def context_vector_from_bits(
    bits: int,
) -> np.ndarray:
    result = np.zeros(
        CONTEXT_COUNT,
        dtype=np.float32,
    )

    for index in range(
        CONTEXT_COUNT
    ):
        if bits & (1 << index):
            result[index] = 1.0

    return result


def context_vector_from_board(
    board: chess.Board,
) -> np.ndarray:
    return context_vector_from_bits(
        context_bits_from_board(
            board
        )
    )


# ============================================================
# FLOAT TRAINING MODEL
# ============================================================

class HalfKP512(nn.Module):
    """
    KnightBot HalfKP-512

        shared 40,960 x 512 feature transformer

             White perspective -> 512
             Black perspective -> 512

        concatenate:
             512 + 512 + 13 context
             = 1037

        1037 -> 64 -> 32 -> 1
    """

    def __init__(self) -> None:
        super().__init__()

        self.feature_weights = nn.EmbeddingBag(
            HALFKP_FEATURE_COUNT,
            TRANSFORMER_HIDDEN,
            mode="sum",
            include_last_offset=False,
        )

        self.transformer_bias = nn.Parameter(
            torch.zeros(
                TRANSFORMER_HIDDEN,
                dtype=torch.float32,
            )
        )

        self.dense1 = nn.Linear(
            TRANSFORMER_HIDDEN * 2 +
            CONTEXT_COUNT,
            DENSE1_COUNT,
        )

        self.dense2 = nn.Linear(
            DENSE1_COUNT,
            DENSE2_COUNT,
        )

        self.output = nn.Linear(
            DENSE2_COUNT,
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
            self.transformer_bias
        )

        nn.init.normal_(
            self.dense1.weight,
            mean=0.0,
            std=0.02,
        )

        nn.init.zeros_(
            self.dense1.bias
        )

        nn.init.normal_(
            self.dense2.weight,
            mean=0.0,
            std=0.02,
        )

        nn.init.zeros_(
            self.dense2.bias
        )

        nn.init.normal_(
            self.output.weight,
            mean=0.0,
            std=0.02,
        )

        nn.init.zeros_(
            self.output.bias
        )

    @staticmethod
    def clipped_activation(
        values: torch.Tensor,
    ) -> torch.Tensor:
        maximum = (
            ACTIVATION_MAX /
            ACTIVATION_SCALE
        )

        return torch.clamp(
            values,
            min=0.0,
            max=maximum,
        )

    def forward(
        self,
        white_features: torch.Tensor,
        black_features: torch.Tensor,
        offsets: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        white = self.feature_weights(
            white_features,
            offsets,
        )

        black = self.feature_weights(
            black_features,
            offsets,
        )

        white = (
            white +
            self.transformer_bias
        )

        black = (
            black +
            self.transformer_bias
        )

        white = self.clipped_activation(
            white
        )

        black = self.clipped_activation(
            black
        )

        combined = torch.cat(
            (
                white,
                black,
                context,
            ),
            dim=1,
        )

        hidden1 = self.dense1(
            combined
        )

        hidden1 = self.clipped_activation(
            hidden1
        )

        hidden2 = self.dense2(
            hidden1
        )

        hidden2 = self.clipped_activation(
            hidden2
        )

        result = self.output(
            hidden2
        )

        return result.squeeze(-1)


# ============================================================
# QUANTIZED MODEL
# ============================================================

@dataclass
class QuantizedHalfKP512:
    transformer_bias: np.ndarray
    transformer_weights: np.ndarray

    dense1_bias: np.ndarray
    dense1_weights: np.ndarray

    dense2_bias: np.ndarray
    dense2_weights: np.ndarray

    output_weights: np.ndarray
    output_bias: int

    activation_max: int
    activation_scale: int
    weight_scale: int
    output_scale: int
    context_active: int


def _clip_int16(
    values: np.ndarray,
) -> np.ndarray:
    return np.clip(
        values,
        np.iinfo(np.int16).min,
        np.iinfo(np.int16).max,
    ).astype(
        np.int16
    )


def _clip_int32(
    values: np.ndarray,
) -> np.ndarray:
    return np.clip(
        values,
        np.iinfo(np.int32).min,
        np.iinfo(np.int32).max,
    ).astype(
        np.int32
    )


def _clip_int32_scalar(
    value: int,
) -> int:
    return int(
        np.clip(
            value,
            np.iinfo(np.int32).min,
            np.iinfo(np.int32).max,
        )
    )


def quantize_model(
    model: HalfKP512,
) -> QuantizedHalfKP512:
    """
    Quantize without moving or changing the live training model.
    """

    with torch.no_grad():
        transformer_weights = (
            model.feature_weights.weight
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        transformer_bias = (
            model.transformer_bias
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        dense1_weights = (
            model.dense1.weight
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        dense1_bias = (
            model.dense1.bias
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        dense2_weights = (
            model.dense2.weight
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        dense2_bias = (
            model.dense2.bias
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

    q_transformer_weights = (
        _clip_int16(
            np.rint(
                transformer_weights *
                ACTIVATION_SCALE
            )
        )
    )

    q_transformer_bias = (
        _clip_int32(
            np.rint(
                transformer_bias *
                ACTIVATION_SCALE
            )
        )
    )

    q_dense1_weights = (
        _clip_int16(
            np.rint(
                dense1_weights *
                WEIGHT_SCALE
            )
        )
    )

    q_dense1_bias = (
        _clip_int32(
            np.rint(
                dense1_bias *
                OUTPUT_SCALE
            )
        )
    )

    q_dense2_weights = (
        _clip_int16(
            np.rint(
                dense2_weights *
                WEIGHT_SCALE
            )
        )
    )

    q_dense2_bias = (
        _clip_int32(
            np.rint(
                dense2_bias *
                OUTPUT_SCALE
            )
        )
    )

    q_output_weights = (
        _clip_int16(
            np.rint(
                output_weights *
                WEIGHT_SCALE
            )
        )
    )

    q_output_bias = (
        _clip_int32_scalar(
            round(
                output_bias *
                OUTPUT_SCALE
            )
        )
    )

    return QuantizedHalfKP512(
        transformer_bias=(
            q_transformer_bias
        ),
        transformer_weights=(
            q_transformer_weights
        ),
        dense1_bias=(
            q_dense1_bias
        ),
        dense1_weights=(
            q_dense1_weights
        ),
        dense2_bias=(
            q_dense2_bias
        ),
        dense2_weights=(
            q_dense2_weights
        ),
        output_weights=(
            q_output_weights
        ),
        output_bias=(
            q_output_bias
        ),
        activation_max=(
            ACTIVATION_MAX
        ),
        activation_scale=(
            ACTIVATION_SCALE
        ),
        weight_scale=(
            WEIGHT_SCALE
        ),
        output_scale=(
            OUTPUT_SCALE
        ),
        context_active=(
            CONTEXT_ACTIVE
        ),
    )


# ============================================================
# BINARY FORMAT
# ============================================================

HEADER_FORMAT = "<IIIIIIiiiii"


def save_quantized_network(
    network: QuantizedHalfKP512,
    path: str | Path,
) -> None:
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected_dense1_inputs = (
        TRANSFORMER_HIDDEN * 2 +
        CONTEXT_COUNT
    )

    if network.transformer_weights.shape != (
        HALFKP_FEATURE_COUNT,
        TRANSFORMER_HIDDEN,
    ):
        raise ValueError(
            "Unexpected transformer weight shape."
        )

    if network.transformer_bias.shape != (
        TRANSFORMER_HIDDEN,
    ):
        raise ValueError(
            "Unexpected transformer bias shape."
        )

    if network.dense1_weights.shape != (
        DENSE1_COUNT,
        expected_dense1_inputs,
    ):
        raise ValueError(
            "Unexpected dense1 weight shape."
        )

    if network.dense1_bias.shape != (
        DENSE1_COUNT,
    ):
        raise ValueError(
            "Unexpected dense1 bias shape."
        )

    if network.dense2_weights.shape != (
        DENSE2_COUNT,
        DENSE1_COUNT,
    ):
        raise ValueError(
            "Unexpected dense2 weight shape."
        )

    if network.dense2_bias.shape != (
        DENSE2_COUNT,
    ):
        raise ValueError(
            "Unexpected dense2 bias shape."
        )

    if network.output_weights.shape != (
        DENSE2_COUNT,
    ):
        raise ValueError(
            "Unexpected output weight shape."
        )

    with path.open(
        "wb"
    ) as output:
        output.write(
            NNUE_MAGIC
        )

        output.write(
            struct.pack(
                HEADER_FORMAT,
                NNUE_FORMAT_VERSION,
                HALFKP_FEATURE_COUNT,
                TRANSFORMER_HIDDEN,
                CONTEXT_COUNT,
                DENSE1_COUNT,
                DENSE2_COUNT,
                network.activation_max,
                network.activation_scale,
                network.weight_scale,
                network.output_scale,
                network.context_active,
            )
        )

        output.write(
            network.transformer_bias
            .astype("<i4")
            .tobytes(order="C")
        )

        output.write(
            network.transformer_weights
            .astype("<i2")
            .tobytes(order="C")
        )

        output.write(
            network.dense1_bias
            .astype("<i4")
            .tobytes(order="C")
        )

        output.write(
            network.dense1_weights
            .astype("<i2")
            .tobytes(order="C")
        )

        output.write(
            network.dense2_bias
            .astype("<i4")
            .tobytes(order="C")
        )

        output.write(
            network.dense2_weights
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
) -> QuantizedHalfKP512:
    path = Path(
        path
    )

    with path.open(
        "rb"
    ) as source:
        magic = source.read(8)

        if magic != NNUE_MAGIC:
            raise ValueError(
                f"Bad HalfKP-512 magic: {magic!r}"
            )

        header_size = struct.calcsize(
            HEADER_FORMAT
        )

        header = source.read(
            header_size
        )

        if len(header) != header_size:
            raise ValueError(
                "HalfKP-512 file ended inside header."
            )

        (
            version,
            feature_count,
            transformer_hidden,
            context_count,
            dense1_count,
            dense2_count,
            activation_max,
            activation_scale,
            weight_scale,
            output_scale,
            context_active,
        ) = struct.unpack(
            HEADER_FORMAT,
            header,
        )

        expected = (
            NNUE_FORMAT_VERSION,
            HALFKP_FEATURE_COUNT,
            TRANSFORMER_HIDDEN,
            CONTEXT_COUNT,
            DENSE1_COUNT,
            DENSE2_COUNT,
        )

        actual = (
            version,
            feature_count,
            transformer_hidden,
            context_count,
            dense1_count,
            dense2_count,
        )

        if actual != expected:
            raise ValueError(
                "HalfKP-512 architecture mismatch: "
                f"{actual} != {expected}"
            )

        transformer_bias = np.frombuffer(
            source.read(
                TRANSFORMER_HIDDEN * 4
            ),
            dtype="<i4",
        ).copy()

        transformer_weights = np.frombuffer(
            source.read(
                HALFKP_FEATURE_COUNT *
                TRANSFORMER_HIDDEN *
                2
            ),
            dtype="<i2",
        ).copy()

        transformer_weights = (
            transformer_weights.reshape(
                HALFKP_FEATURE_COUNT,
                TRANSFORMER_HIDDEN,
            )
        )

        dense1_bias = np.frombuffer(
            source.read(
                DENSE1_COUNT * 4
            ),
            dtype="<i4",
        ).copy()

        dense1_inputs = (
            TRANSFORMER_HIDDEN * 2 +
            CONTEXT_COUNT
        )

        dense1_weights = np.frombuffer(
            source.read(
                DENSE1_COUNT *
                dense1_inputs *
                2
            ),
            dtype="<i2",
        ).copy()

        dense1_weights = (
            dense1_weights.reshape(
                DENSE1_COUNT,
                dense1_inputs,
            )
        )

        dense2_bias = np.frombuffer(
            source.read(
                DENSE2_COUNT * 4
            ),
            dtype="<i4",
        ).copy()

        dense2_weights = np.frombuffer(
            source.read(
                DENSE2_COUNT *
                DENSE1_COUNT *
                2
            ),
            dtype="<i2",
        ).copy()

        dense2_weights = (
            dense2_weights.reshape(
                DENSE2_COUNT,
                DENSE1_COUNT,
            )
        )

        output_weights = np.frombuffer(
            source.read(
                DENSE2_COUNT * 2
            ),
            dtype="<i2",
        ).copy()

        output_bias_raw = source.read(
            4
        )

        if len(output_bias_raw) != 4:
            raise ValueError(
                "HalfKP-512 file ended before output bias."
            )

        output_bias = struct.unpack(
            "<i",
            output_bias_raw,
        )[0]

        if source.read(1):
            raise ValueError(
                "Unexpected trailing data in HalfKP-512 file."
            )

    return QuantizedHalfKP512(
        transformer_bias=transformer_bias,
        transformer_weights=transformer_weights,
        dense1_bias=dense1_bias,
        dense1_weights=dense1_weights,
        dense2_bias=dense2_bias,
        dense2_weights=dense2_weights,
        output_weights=output_weights,
        output_bias=output_bias,
        activation_max=activation_max,
        activation_scale=activation_scale,
        weight_scale=weight_scale,
        output_scale=output_scale,
        context_active=context_active,
    )


# ============================================================
# INTEGER INFERENCE
# ============================================================

def _dense_integer_layer(
    inputs: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    activation_max: int,
    weight_scale: int,
) -> np.ndarray:
    raw = (
        bias.astype(np.int64)
        +
        weights.astype(np.int64)
        @
        inputs.astype(np.int64)
    )

    # Convert from activation_scale * weight_scale
    # back to activation_scale.
    if weight_scale <= 0:
        raise ValueError(
            "Invalid weight scale."
        )

    positive = raw >= 0

    scaled = np.empty_like(
        raw,
        dtype=np.int64,
    )

    scaled[positive] = (
        raw[positive] //
        weight_scale
    )

    scaled[~positive] = -(
        (-raw[~positive]) //
        weight_scale
    )

    return np.clip(
        scaled,
        0,
        activation_max,
    )


def evaluate_quantized_board(
    network: QuantizedHalfKP512,
    board: chess.Board,
) -> int:
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

    white_accumulator = (
        network.transformer_bias
        .astype(np.int64)
        .copy()
    )

    black_accumulator = (
        network.transformer_bias
        .astype(np.int64)
        .copy()
    )

    for feature in white_features:
        white_accumulator += (
            network.transformer_weights[
                feature
            ].astype(np.int64)
        )

    for feature in black_features:
        black_accumulator += (
            network.transformer_weights[
                feature
            ].astype(np.int64)
        )

    white_accumulator = np.clip(
        white_accumulator,
        0,
        network.activation_max,
    )

    black_accumulator = np.clip(
        black_accumulator,
        0,
        network.activation_max,
    )

    context_bits = context_bits_from_board(
        board
    )

    context = np.zeros(
        CONTEXT_COUNT,
        dtype=np.int64,
    )

    for index in range(
        CONTEXT_COUNT
    ):
        if context_bits & (1 << index):
            context[index] = (
                network.context_active
            )

    combined = np.concatenate(
        (
            white_accumulator,
            black_accumulator,
            context,
        )
    )

    hidden1 = _dense_integer_layer(
        combined,
        network.dense1_weights,
        network.dense1_bias,
        network.activation_max,
        network.weight_scale,
    )

    hidden2 = _dense_integer_layer(
        hidden1,
        network.dense2_weights,
        network.dense2_bias,
        network.activation_max,
        network.weight_scale,
    )

    raw_output = int(
        network.output_bias
    )

    raw_output += int(
        np.sum(
            hidden2.astype(np.int64)
            *
            network.output_weights.astype(
                np.int64
            )
        )
    )

    if raw_output >= 0:
        cp = (
            raw_output //
            network.output_scale
        )
    else:
        cp = -(
            (-raw_output) //
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
    network: QuantizedHalfKP512,
    fen: str,
) -> int:
    return evaluate_quantized_board(
        network,
        chess.Board(
            fen
        ),
    )