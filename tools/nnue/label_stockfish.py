from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import chess
import chess.engine
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
LABELS_DIR = DATA_DIR / "labels"


# ============================================================
# WORKER GLOBALS
# ============================================================

_worker_engine: Optional[
    chess.engine.SimpleEngine
] = None

_worker_limit: Optional[
    chess.engine.Limit
] = None

_worker_mate_score: int = 30000
_worker_target_clamp: int = 3000


# ============================================================
# COMMAND LINE
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel Stockfish teacher labeler "
            "for KnightBot NNUE training.\n\n"
            "If --output is omitted, labels are automatically "
            "stored under data/labels/<name>/training.tsv."
        )
    )

    parser.add_argument(
        "--stockfish",
        required=True,
        help="Path to Stockfish executable.",
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input FEN file.",
    )

    parser.add_argument(
        "--output",
        default="",
        help=(
            "Explicit output TSV path. "
            "If omitted, uses "
            "data/labels/<name>/training.tsv."
        ),
    )

    parser.add_argument(
        "--name",
        default="training",
        help=(
            "Dataset/run name used when --output "
            "is omitted. Default: training."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Number of parallel Stockfish processes. "
            "0 = automatic."
        ),
    )

    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=1,
        help=(
            "Stockfish Threads setting for each worker. "
            "Default: 1."
        ),
    )

    parser.add_argument(
        "--hash-per-worker",
        type=int,
        default=64,
        help=(
            "Stockfish hash in MB for each worker. "
            "Default: 64."
        ),
    )

    search_group = (
        parser.add_mutually_exclusive_group()
    )

    search_group.add_argument(
        "--depth",
        type=int,
        default=12,
        help=(
            "Stockfish search depth. "
            "Default: 12."
        ),
    )

    search_group.add_argument(
        "--nodes",
        type=int,
        default=None,
        help=(
            "Use a fixed node count per position "
            "instead of fixed depth."
        ),
    )

    parser.add_argument(
        "--mate-score",
        type=int,
        default=30000,
        help=(
            "CP value used for mate scores. "
            "Default: 30000."
        ),
    )

    parser.add_argument(
        "--target-clamp",
        type=int,
        default=3000,
        help=(
            "Clamp Stockfish labels to +/- this "
            "centipawn value. Default: 3000."
        ),
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=8,
        help=(
            "Multiprocessing work chunk size. "
            "Default: 8."
        ),
    )

    parser.add_argument(
        "--flush-every",
        type=int,
        default=250,
        help=(
            "Flush output and checkpoint every N "
            "processed input positions. Default: 250."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume using the sidecar progress "
            "checkpoint from a previous run."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Maximum number of input positions to "
            "process in this run. 0 = unlimited."
        ),
    )

    return parser.parse_args()


# ============================================================
# OUTPUT ORGANIZATION
# ============================================================

def resolve_output_path(
    explicit_output: str,
    name: str,
) -> Path:
    if explicit_output:
        return Path(
            explicit_output
        )

    safe_name = name.strip()

    if not safe_name:
        safe_name = "training"

    run_dir = (
        LABELS_DIR /
        safe_name
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        run_dir /
        "training.tsv"
    )


def metadata_path_for(
    output_path: Path,
) -> Path:
    return (
        output_path.parent /
        (
            output_path.stem +
            ".meta.json"
        )
    )


def write_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    stockfish_path: Path,
    input_path: Path,
    output_path: Path,
    worker_count: int,
    total_input: int,
    status: str,
    processed: int = 0,
    written: int = 0,
    terminal_skipped: int = 0,
    invalid_skipped: int = 0,
    engine_errors: int = 0,
    next_input_index: int = 0,
) -> None:
    if args.nodes is not None:
        search = {
            "type": "nodes",
            "value": args.nodes,
        }

    else:
        search = {
            "type": "depth",
            "value": args.depth,
        }

    metadata = {
        "format_version": 1,
        "status": status,
        "updated": datetime.now().isoformat(
            timespec="seconds"
        ),
        "stockfish": str(
            stockfish_path.resolve()
        ),
        "input": str(
            input_path.resolve()
        ),
        "output": str(
            output_path.resolve()
        ),
        "total_input_positions": total_input,
        "search": search,
        "workers": worker_count,
        "threads_per_worker": (
            args.threads_per_worker
        ),
        "hash_per_worker_mb": (
            args.hash_per_worker
        ),
        "mate_score": (
            args.mate_score
        ),
        "target_clamp_cp": (
            args.target_clamp
        ),
        "chunksize": (
            args.chunksize
        ),
        "flush_every": (
            args.flush_every
        ),
        "processed_this_run": processed,
        "labels_written_this_run": written,
        "terminal_skipped_this_run": (
            terminal_skipped
        ),
        "invalid_skipped_this_run": (
            invalid_skipped
        ),
        "engine_errors_this_run": (
            engine_errors
        ),
        "next_input_index": (
            next_input_index
        ),
    }

    temporary = Path(
        str(path) +
        ".tmp"
    )

    temporary.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        +
        "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


# ============================================================
# FILE HELPERS
# ============================================================

def count_lines(
    path: Path,
) -> int:
    count = 0

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as source:
        for _ in source:
            count += 1

    return count


def checkpoint_path_for(
    output_path: Path,
) -> Path:
    return Path(
        str(output_path) +
        ".progress"
    )


def read_checkpoint(
    path: Path,
) -> int:
    if not path.exists():
        return 0

    try:
        text = path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            return 0

        value = int(
            text
        )

        return max(
            0,
            value,
        )

    except (
        ValueError,
        OSError,
    ):
        return 0


def write_checkpoint_atomic(
    path: Path,
    next_input_index: int,
) -> None:
    temporary = Path(
        str(path) +
        ".tmp"
    )

    temporary.write_text(
        str(
            next_input_index
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


# ============================================================
# STOCKFISH WORKER
# ============================================================

def close_worker_engine() -> None:
    global _worker_engine

    if _worker_engine is not None:
        try:
            _worker_engine.quit()

        except Exception:
            pass

        _worker_engine = None


def initialize_worker(
    stockfish_path: str,
    threads_per_worker: int,
    hash_per_worker: int,
    depth: Optional[int],
    nodes: Optional[int],
    mate_score: int,
    target_clamp: int,
) -> None:
    global _worker_engine
    global _worker_limit
    global _worker_mate_score
    global _worker_target_clamp

    _worker_engine = (
        chess.engine.SimpleEngine.popen_uci(
            stockfish_path
        )
    )

    _worker_engine.configure(
        {
            "Threads": (
                threads_per_worker
            ),
            "Hash": (
                hash_per_worker
            ),
        }
    )

    if nodes is not None:
        _worker_limit = (
            chess.engine.Limit(
                nodes=nodes
            )
        )

    else:
        _worker_limit = (
            chess.engine.Limit(
                depth=depth
            )
        )

    _worker_mate_score = (
        mate_score
    )

    _worker_target_clamp = (
        target_clamp
    )


# ============================================================
# LABEL ONE POSITION
# ============================================================

def label_position(
    item: tuple[
        int,
        str,
    ]
) -> tuple[
    int,
    str,
    Optional[int],
    Optional[str],
]:
    input_index, fen = item

    fen = fen.strip()

    if not fen:
        return (
            input_index,
            fen,
            None,
            "empty",
        )

    try:
        board = chess.Board(
            fen
        )

    except ValueError as error:
        return (
            input_index,
            fen,
            None,
            f"invalid FEN: {error}",
        )

    if board.is_game_over(
        claim_draw=True
    ):
        return (
            input_index,
            fen,
            None,
            "terminal",
        )

    if (
        _worker_engine is None
        or
        _worker_limit is None
    ):
        return (
            input_index,
            fen,
            None,
            "worker not initialized",
        )

    try:
        info = (
            _worker_engine.analyse(
                board,
                _worker_limit,
            )
        )

        score_object = (
            info["score"]
            .pov(
                chess.WHITE
            )
        )

        score = (
            score_object.score(
                mate_score=(
                    _worker_mate_score
                )
            )
        )

        if score is None:
            return (
                input_index,
                fen,
                None,
                "no score",
            )

        score = max(
            -_worker_target_clamp,
            min(
                _worker_target_clamp,
                int(
                    score
                ),
            ),
        )

        return (
            input_index,
            fen,
            score,
            None,
        )

    except Exception as error:
        return (
            input_index,
            fen,
            None,
            f"engine error: {error}",
        )


# ============================================================
# INPUT GENERATOR
# ============================================================

def input_positions(
    path: Path,
    start_index: int,
    limit: int,
):
    emitted = 0

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as source:
        for input_index, line in enumerate(
            source
        ):
            if (
                input_index <
                start_index
            ):
                continue

            if (
                limit > 0
                and
                emitted >= limit
            ):
                break

            yield (
                input_index,
                line.rstrip(
                    "\r\n"
                ),
            )

            emitted += 1


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    stockfish_path = Path(
        args.stockfish
    )

    input_path = Path(
        args.input
    )

    output_path = resolve_output_path(
        args.output,
        args.name,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    progress_path = (
        checkpoint_path_for(
            output_path
        )
    )

    metadata_path = (
        metadata_path_for(
            output_path
        )
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    if not stockfish_path.exists():
        raise FileNotFoundError(
            f"Stockfish not found: "
            f"{stockfish_path}"
        )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: "
            f"{input_path}"
        )

    if args.threads_per_worker < 1:
        raise ValueError(
            "--threads-per-worker "
            "must be at least 1."
        )

    if args.hash_per_worker < 1:
        raise ValueError(
            "--hash-per-worker "
            "must be at least 1."
        )

    if args.chunksize < 1:
        raise ValueError(
            "--chunksize must be "
            "at least 1."
        )

    if args.flush_every < 1:
        raise ValueError(
            "--flush-every must be "
            "at least 1."
        )

    # ========================================================
    # WORKER COUNT
    # ========================================================

    logical_cpus = (
        os.cpu_count()
        or
        4
    )

    if args.workers > 0:
        worker_count = (
            args.workers
        )

    else:
        worker_count = max(
            1,
            logical_cpus - 2,
        )

    maximum_workers = max(
        1,
        logical_cpus
        //
        args.threads_per_worker,
    )

    worker_count = min(
        worker_count,
        maximum_workers,
    )

    # ========================================================
    # RESUME
    # ========================================================

    start_index = 0

    if args.resume:
        if not progress_path.exists():
            print(
                "ERROR: --resume was requested, "
                "but no progress checkpoint exists:"
            )

            print(
                progress_path
            )

            print(
                "\nThe labeler does not guess the "
                "resume position from TSV line count."
            )

            sys.exit(
                1
            )

        start_index = (
            read_checkpoint(
                progress_path
            )
        )

        output_mode = "a"

    else:
        output_mode = "w"

        if progress_path.exists():
            progress_path.unlink()

    # ========================================================
    # COUNT INPUT
    # ========================================================

    print(
        "Counting input positions..."
    )

    total_input = (
        count_lines(
            input_path
        )
    )

    remaining = max(
        0,
        total_input -
        start_index,
    )

    if args.limit > 0:
        remaining = min(
            remaining,
            args.limit,
        )

    if remaining == 0:
        print(
            "No positions left to process."
        )

        return

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        "KnightBot parallel Stockfish labeler"
    )
    print(
        "========================================"
    )
    print(
        f"Input:              {input_path}"
    )
    print(
        f"Output:             {output_path}"
    )
    print(
        f"Metadata:           {metadata_path}"
    )
    print(
        f"Total input:        {total_input:,}"
    )
    print(
        f"Starting index:     {start_index:,}"
    )
    print(
        f"Positions this run: {remaining:,}"
    )
    print(
        f"Workers:            {worker_count}"
    )
    print(
        "Threads / worker:   "
        f"{args.threads_per_worker}"
    )
    print(
        "Hash / worker:      "
        f"{args.hash_per_worker} MB"
    )

    if args.nodes is not None:
        print(
            f"Search:             "
            f"{args.nodes:,} nodes"
        )

    else:
        print(
            f"Search:             "
            f"depth {args.depth}"
        )

    print(
        f"Target clamp:       "
        f"+/-{args.target_clamp} cp"
    )
    print(
        f"Resume checkpoint:  "
        f"{progress_path}"
    )
    print(
        "========================================"
    )
    print()

    write_metadata(
        metadata_path,
        args=args,
        stockfish_path=stockfish_path,
        input_path=input_path,
        output_path=output_path,
        worker_count=worker_count,
        total_input=total_input,
        status="running",
        next_input_index=start_index,
    )

    # ========================================================
    # PROCESS
    # ========================================================

    processed = 0
    written = 0
    skipped_terminal = 0
    skipped_invalid = 0
    errors = 0

    last_completed_index = (
        start_index
    )

    pool = Pool(
        processes=worker_count,
        initializer=initialize_worker,
        initargs=(
            str(
                stockfish_path
            ),
            args.threads_per_worker,
            args.hash_per_worker,
            args.depth,
            args.nodes,
            args.mate_score,
            args.target_clamp,
        ),
    )

    try:
        with output_path.open(
            output_mode,
            encoding="utf-8",
            newline="\n",
            buffering=1024 * 1024,
        ) as output:
            try:
                jobs = input_positions(
                    input_path,
                    start_index,
                    args.limit,
                )

                results = pool.imap(
                    label_position,
                    jobs,
                    chunksize=args.chunksize,
                )

                progress = tqdm(
                    results,
                    total=remaining,
                    desc="Stockfish labeling",
                    unit="pos",
                    dynamic_ncols=True,
                )

                for (
                    input_index,
                    fen,
                    score,
                    error,
                ) in progress:
                    processed += 1

                    last_completed_index = (
                        input_index +
                        1
                    )

                    if score is not None:
                        output.write(
                            f"{fen}\t{score}\n"
                        )

                        written += 1

                    elif error == "terminal":
                        skipped_terminal += 1

                    elif (
                        error == "empty"
                        or
                        (
                            error is not None
                            and
                            error.startswith(
                                "invalid FEN"
                            )
                        )
                    ):
                        skipped_invalid += 1

                    else:
                        errors += 1

                        if error is not None:
                            tqdm.write(
                                f"Warning at input "
                                f"{input_index}: "
                                f"{error}"
                            )

                    if (
                        processed
                        %
                        args.flush_every
                        ==
                        0
                    ):
                        output.flush()

                        try:
                            os.fsync(
                                output.fileno()
                            )
                        except OSError:
                            pass

                        write_checkpoint_atomic(
                            progress_path,
                            last_completed_index,
                        )

                    progress.set_postfix(
                        saved=f"{written:,}",
                        errors=errors,
                    )

                output.flush()

                try:
                    os.fsync(
                        output.fileno()
                    )
                except OSError:
                    pass

                write_checkpoint_atomic(
                    progress_path,
                    last_completed_index,
                )

            except KeyboardInterrupt:
                # Crucial resume-safety behavior:
                #
                # Flush every label that has already been written, then
                # advance the checkpoint to exactly the last consumed
                # input position. This prevents duplicate rows when the
                # run is resumed after Ctrl+C.
                output.flush()

                try:
                    os.fsync(
                        output.fileno()
                    )
                except OSError:
                    pass

                write_checkpoint_atomic(
                    progress_path,
                    last_completed_index,
                )

                raise

    except KeyboardInterrupt:
        print()
        print(
            "Ctrl+C received."
        )
        print(
            "Stopping workers..."
        )

        pool.terminate()
        pool.join()

        write_metadata(
            metadata_path,
            args=args,
            stockfish_path=stockfish_path,
            input_path=input_path,
            output_path=output_path,
            worker_count=worker_count,
            total_input=total_input,
            status="interrupted",
            processed=processed,
            written=written,
            terminal_skipped=(
                skipped_terminal
            ),
            invalid_skipped=(
                skipped_invalid
            ),
            engine_errors=errors,
            next_input_index=(
                last_completed_index
            ),
        )

        print(
            "Output and checkpoint were flushed."
        )
        print(
            "Resume this exact run with --resume."
        )
        print(
            f"Next input index: "
            f"{last_completed_index:,}"
        )

        return

    except Exception:
        pool.terminate()
        pool.join()

        write_metadata(
            metadata_path,
            args=args,
            stockfish_path=stockfish_path,
            input_path=input_path,
            output_path=output_path,
            worker_count=worker_count,
            total_input=total_input,
            status="error",
            processed=processed,
            written=written,
            terminal_skipped=(
                skipped_terminal
            ),
            invalid_skipped=(
                skipped_invalid
            ),
            engine_errors=errors,
            next_input_index=(
                last_completed_index
            ),
        )

        raise

    else:
        # All results have been consumed and output/checkpoint are
        # already safely flushed. Force termination avoids the
        # python-chess SimpleEngine worker shutdown hang on Windows.
        pool.terminate()
        pool.join()

    write_metadata(
        metadata_path,
        args=args,
        stockfish_path=stockfish_path,
        input_path=input_path,
        output_path=output_path,
        worker_count=worker_count,
        total_input=total_input,
        status="complete",
        processed=processed,
        written=written,
        terminal_skipped=(
            skipped_terminal
        ),
        invalid_skipped=(
            skipped_invalid
        ),
        engine_errors=errors,
        next_input_index=(
            last_completed_index
        ),
    )

    # ========================================================
    # RESULT
    # ========================================================

    print()
    print(
        "Labeling complete"
    )
    print(
        "========================================"
    )
    print(
        f"Input processed:    "
        f"{processed:,}"
    )
    print(
        f"Labels written:     "
        f"{written:,}"
    )
    print(
        f"Terminal skipped:   "
        f"{skipped_terminal:,}"
    )
    print(
        f"Invalid skipped:    "
        f"{skipped_invalid:,}"
    )
    print(
        f"Engine errors:      "
        f"{errors:,}"
    )
    print(
        f"Next input index:   "
        f"{last_completed_index:,}"
    )
    print(
        f"Output:             "
        f"{output_path}"
    )
    print(
        f"Checkpoint:         "
        f"{progress_path}"
    )
    print(
        f"Metadata:           "
        f"{metadata_path}"
    )
    print(
        "========================================"
    )


if __name__ == "__main__":
    # Required for multiprocessing on Windows.
    main()