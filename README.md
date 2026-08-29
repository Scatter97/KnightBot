# KnightBot

KnightBot is a custom C++ chess engine built from the ground up as a learning and experimentation project.

Current version: **v0.10.0**

KnightBot supports UCI, modern alpha-beta search techniques, handcrafted evaluation, and NNUE evaluation using HalfKP features.

## Features

### Search

KnightBot currently includes:

- Iterative deepening
- Negamax alpha-beta search
- Principal Variation Search
- Aspiration windows
- Transposition table
- Killer move heuristic
- History heuristic
- Static Exchange Evaluation
- Quiescence search
- Null-move pruning
- Late Move Reductions
- Check extensions
- Razoring
- Reverse futility pruning
- Move futility pruning
- Repetition detection
- Fifty-move rule handling
- Insufficient-material detection

### Move generation

- Legal move generation
- Castling
- En passant
- Promotions
- Incremental board state
- Bitboard-based move generation
- Zobrist hashing

### Evaluation

KnightBot supports:

- Handcrafted evaluation
- NNUE evaluation

The current NNUE architecture uses:

- HalfKP-style features
- 40,960 HalfKP feature inputs
- Two 512-unit accumulators
- Side-to-move and position context features
- Incremental accumulator updates
- Lazy accumulator materialization
- Integer inference optimized for AVX2

Trained NNUE weights are not currently included in the repository.

## Performance

KnightBot includes a deterministic regression benchmark.

For v0.10.0:

```text
Positions: 6
Nodes:     157574
Checksum:  5907223103162280312
```

A recent Release build using the HalfKP NNUE evaluator achieved roughly:

```text
200,000+ nodes per second
```

Actual performance depends on CPU, compiler, build configuration, and NNUE network.

KnightBot also includes:

```text
benchfull
```

which reports detailed profiling information including:

- QSearch nodes
- QSearch node share
- TT probes, hits, and cutoffs
- Evaluation-cache hits and misses
- Null-move attempts and cutoffs
- Razoring
- Reverse futility pruning
- Move futility pruning
- Late Move Reductions
- LMR re-searches
- PVS re-searches
- Check extensions
- HalfKP materializations
- Lazy HalfKP skips

## Building

KnightBot uses CMake and requires a C++20-capable compiler.

The current Windows development environment uses Visual Studio with MSVC.

Open the project folder in Visual Studio and build the **Release** configuration.

Release builds currently use AVX2 optimizations.

## Running

Launch the KnightBot executable.

Useful CLI commands include:

```text
board
moves
move <uci>
eval
evaldetail
evalcompare
nnue on
nnue off
nnueload <file>
nnueinfo
nnueverify
selftest
bench
benchfull
bestmove <depth>
think <milliseconds>
perft <depth>
divide <depth>
startpos
fen <FEN string>
showfen
ttclear
ttstats
help
quit
```

## UCI

KnightBot supports the Universal Chess Interface.

Example:

```text
uci
isready
ucinewgame
position startpos
go depth 6
```

NNUE can also be configured through UCI:

```text
setoption name UseNNUE value true
setoption name EvalFile value <path>
```

## NNUE Training Tools

Training and verification tools are located under:

```text
tools/nnue/
```

These include tools for:

- Extracting positions from PGN files
- Labeling positions with Stockfish
- Dataset validation
- Dataset pre-encoding
- NNUE training
- HalfKP-512 training
- Network export
- Quantization validation
- Python/C++ evaluation comparison

Generated datasets, checkpoints, and training runs are intentionally excluded from Git.

## Testing

Useful development commands:

```text
selftest
bench
benchfull
nnueverify
```

Known start-position perft values:

```text
Depth 1: 20
Depth 2: 400
Depth 3: 8902
Depth 4: 197281
Depth 5: 4865609
Depth 6: 119060324
Depth 7: 3195901860
Depth 8: 84998978956
Depth 9: 2439530234167
```

Kiwipete:

```text
r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1
```

Reference values:

```text
Depth 1: 48
Depth 2: 2039
Depth 3: 97862
Depth 4: 4085603
```

## Project Structure

```text
src/
    chess.cpp
    chess.hpp
    evaluation.cpp
    evaluation.hpp
    search.cpp
    search.hpp
    nnue.cpp
    nnue.hpp
    devtools.cpp
    devtools.hpp
    training.cpp
    training.hpp
    main.cpp

tools/
    nnue/
```

## Development Status

KnightBot is experimental and actively being developed.

Current areas of focus include:

- Search strength
- NNUE evaluation quality
- NNUE inference speed
- Quiescence-search efficiency
- Benchmarking and profiling
- Engine strength testing

## Version

**KnightBot v0.10.0**

## CopyRight

Copyright © 2026 Joshua Wang
All rights reserved.
This software and its source code may not be copied, modified, distributed, sublicensed, or used for commercial purposes without prior written permission from the copyright holder.
