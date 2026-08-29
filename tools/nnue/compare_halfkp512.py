from __future__ import annotations

import argparse

import chess
import numpy as np
import torch

from halfkp_512 import (
    CONTEXT_COUNT,
    HalfKP512,
    context_bits_from_board,
    evaluate_quantized_board,
    halfkp_features_from_board,
    load_quantized_network,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--network",
        required=True,
    )

    parser.add_argument(
        "--fen",
        required=True,
    )

    return parser.parse_args()


def make_float_inputs(board, device):
    white = halfkp_features_from_board(
        board,
        chess.WHITE,
    )

    black = halfkp_features_from_board(
        board,
        chess.BLACK,
    )

    white_tensor = torch.tensor(
        white,
        dtype=torch.long,
        device=device,
    )

    black_tensor = torch.tensor(
        black,
        dtype=torch.long,
        device=device,
    )

    offsets = torch.tensor(
        [0],
        dtype=torch.long,
        device=device,
    )

    context = np.zeros(
        (1, CONTEXT_COUNT),
        dtype=np.float32,
    )

    bits = context_bits_from_board(
        board
    )

    for index in range(CONTEXT_COUNT):
        if bits & (1 << index):
            context[0, index] = 1.0

    context_tensor = torch.from_numpy(
        context
    ).to(device)

    return (
        white_tensor,
        black_tensor,
        offsets,
        context_tensor,
    )


def main():
    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model = HalfKP512().to(device)

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    board = chess.Board(
        args.fen
    )

    (
        white,
        black,
        offsets,
        context,
    ) = make_float_inputs(
        board,
        device,
    )

    with torch.no_grad():
        float_score = model(
            white,
            black,
            offsets,
            context,
        ).item()

    quantized = load_quantized_network(
        args.network
    )

    quantized_score = evaluate_quantized_board(
        quantized,
        board,
    )

    print(
        f"Checkpoint epoch: {checkpoint['epoch']}"
    )

    print(
        f"Float score:      {float_score:.3f} cp"
    )

    print(
        f"Quantized score:  {quantized_score} cp"
    )

    print(
        f"Difference:       "
        f"{quantized_score - float_score:+.3f} cp"
    )


if __name__ == "__main__":
    main()