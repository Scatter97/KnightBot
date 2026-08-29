from __future__ import annotations

import argparse
import io
import random
from pathlib import Path
from typing import TextIO

import chess
import chess.pgn
import zstandard as zstd
from tqdm import tqdm

from knightbot_nnue import normalized_position_key


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
POSITIONS_DIR = DATA_DIR / "positions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract deduplicated chess positions from PGN files "
            "for KnightBot NNUE training.\n\n"
            "Supported inputs:\n"
            "  .pgn\n"
            "  .pgn.zst\n\n"
            "If --output is not supplied, output is automatically "
            "placed under data/positions/."
        )
    )

    parser.add_argument(
        "pgn",
        nargs="+",
        help="Input .pgn or .pgn.zst file(s).",
    )

    parser.add_argument(
        "--output",
        default="",
        help=(
            "Explicit output FEN path. "
            "If omitted, uses data/positions/<name>.fen."
        ),
    )

    parser.add_argument(
        "--name",
        default="positions",
        help=(
            "Dataset name used when --output is omitted. "
            "Default: positions."
        ),
    )

    parser.add_argument(
        "--skip-opening-plies",
        type=int,
        default=8,
        help=(
            "Ignore the first N plies of each game. "
            "Default: 8."
        ),
    )

    parser.add_argument(
        "--every",
        type=int,
        default=2,
        help=(
            "Consider one position every N plies. "
            "Default: 2."
        ),
    )

    parser.add_argument(
        "--keep-probability",
        type=float,
        default=1.0,
        help=(
            "Random probability of retaining an eligible "
            "position. Default: 1.0."
        ),
    )

    parser.add_argument(
        "--max-positions",
        type=int,
        default=0,
        help=(
            "Stop after this many unique positions. "
            "0 = unlimited."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    return parser.parse_args()


def resolve_output_path(
    explicit_output: str,
    name: str,
) -> Path:
    if explicit_output:
        return Path(explicit_output)

    safe_name = name.strip()

    if not safe_name:
        safe_name = "positions"

    if not safe_name.lower().endswith(".fen"):
        safe_name += ".fen"

    POSITIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return POSITIONS_DIR / safe_name


class PgnTextStream:
    """
    Context manager that opens either:

        plain .pgn
        compressed .pgn.zst

    as a normal text stream suitable for python-chess.

    The .zst file is decompressed incrementally. It is never fully
    unpacked to disk or loaded into RAM.
    """

    def __init__(
        self,
        path: Path,
    ) -> None:
        self.path = path

        self._plain_file = None
        self._compressed_file = None
        self._reader = None
        self._buffered_reader = None
        self._text_wrapper = None

    def __enter__(
        self,
    ) -> TextIO:
        lower_name = self.path.name.lower()

        if lower_name.endswith(
            ".pgn.zst"
        ):
            self._compressed_file = (
                self.path.open(
                    "rb"
                )
            )

            decompressor = (
                zstd.ZstdDecompressor()
            )

            self._reader = (
                decompressor.stream_reader(
                    self._compressed_file
                )
            )

            self._buffered_reader = (
                io.BufferedReader(
                    self._reader,
                    buffer_size=1024 * 1024,
                )
            )

            self._text_wrapper = (
                io.TextIOWrapper(
                    self._buffered_reader,
                    encoding="utf-8",
                    errors="replace",
                    newline=None,
                )
            )

            return self._text_wrapper

        self._plain_file = self.path.open(
            "r",
            encoding="utf-8",
            errors="replace",
            newline=None,
        )

        return self._plain_file

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        if self._text_wrapper is not None:
            try:
                self._text_wrapper.close()
            except Exception:
                pass

        elif self._buffered_reader is not None:
            try:
                self._buffered_reader.close()
            except Exception:
                pass

        elif self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass

        if self._plain_file is not None:
            try:
                self._plain_file.close()
            except Exception:
                pass

        if self._compressed_file is not None:
            try:
                self._compressed_file.close()
            except Exception:
                pass


def open_pgn_text(
    path: Path,
) -> PgnTextStream:
    return PgnTextStream(
        path
    )


def main() -> None:
    args = parse_args()

    if args.every <= 0:
        raise ValueError(
            "--every must be at least 1."
        )

    if args.skip_opening_plies < 0:
        raise ValueError(
            "--skip-opening-plies cannot be negative."
        )

    if not (
        0.0 <
        args.keep_probability <=
        1.0
    ):
        raise ValueError(
            "--keep-probability must be in (0, 1]."
        )

    if args.max_positions < 0:
        raise ValueError(
            "--max-positions cannot be negative."
        )

    random.seed(
        args.seed
    )

    seen: set[str] = set()

    written = 0
    games = 0
    malformed_games = 0

    output_path = resolve_output_path(
        args.output,
        args.name,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Output positions: {output_path}"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
        buffering=1024 * 1024,
    ) as output:
        for pgn_name in args.pgn:
            pgn_path = Path(
                pgn_name
            )

            if not pgn_path.exists():
                raise FileNotFoundError(
                    pgn_path
                )

            lower_name = (
                pgn_path.name.lower()
            )

            if not (
                lower_name.endswith(
                    ".pgn"
                )
                or
                lower_name.endswith(
                    ".pgn.zst"
                )
            ):
                raise ValueError(
                    "Unsupported PGN input type: "
                    f"{pgn_path}\n"
                    "Expected .pgn or .pgn.zst"
                )

            print()
            print(
                f"Reading {pgn_path}"
            )

            if lower_name.endswith(
                ".pgn.zst"
            ):
                print(
                    "Streaming zstd-compressed PGN "
                    "without extracting to disk."
                )

            with open_pgn_text(
                pgn_path
            ) as source:
                progress = tqdm(
                    desc=pgn_path.name,
                    unit="games",
                    dynamic_ncols=True,
                )

                while True:
                    try:
                        game = (
                            chess.pgn.read_game(
                                source
                            )
                        )

                    except Exception as error:
                        malformed_games += 1

                        tqdm.write(
                            "Warning: unable to parse "
                            f"PGN game: {error}"
                        )

                        continue

                    if game is None:
                        break

                    games += 1
                    progress.update(1)

                    board = game.board()

                    try:
                        mainline_moves = (
                            game.mainline_moves()
                        )

                        for ply, move in enumerate(
                            mainline_moves,
                            start=1,
                        ):
                            if (
                                move not in
                                board.legal_moves
                            ):
                                malformed_games += 1
                                break

                            board.push(
                                move
                            )

                            if (
                                ply <=
                                args.skip_opening_plies
                            ):
                                continue

                            if (
                                (
                                    ply -
                                    args.skip_opening_plies
                                )
                                %
                                args.every
                                !=
                                0
                            ):
                                continue

                            if (
                                random.random()
                                >
                                args.keep_probability
                            ):
                                continue

                            if board.is_game_over(
                                claim_draw=True
                            ):
                                continue

                            key = (
                                normalized_position_key(
                                    board
                                )
                            )

                            if key in seen:
                                continue

                            seen.add(
                                key
                            )

                            output.write(
                                board.fen(
                                    en_passant="fen"
                                )
                                +
                                "\n"
                            )

                            written += 1

                            if (
                                written %
                                10000
                                ==
                                0
                            ):
                                output.flush()

                                progress.set_postfix(
                                    positions=f"{written:,}",
                                    unique=f"{len(seen):,}",
                                )

                            if (
                                args.max_positions > 0
                                and
                                written >=
                                args.max_positions
                            ):
                                output.flush()
                                progress.close()

                                print()
                                print(
                                    "Reached position limit."
                                )
                                print(
                                    f"Games read: "
                                    f"{games:,}"
                                )
                                print(
                                    "Unique positions: "
                                    f"{written:,}"
                                )
                                print(
                                    "Malformed games: "
                                    f"{malformed_games:,}"
                                )
                                print(
                                    f"Saved to: "
                                    f"{output_path}"
                                )

                                return

                    except Exception as error:
                        malformed_games += 1

                        tqdm.write(
                            "Warning while processing "
                            f"game {games}: {error}"
                        )

                        continue

                progress.close()

    print()
    print(
        f"Games read: "
        f"{games:,}"
    )
    print(
        f"Unique positions: "
        f"{written:,}"
    )
    print(
        f"Malformed games: "
        f"{malformed_games:,}"
    )
    print(
        f"Saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()