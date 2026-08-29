#pragma once

#include "chess.hpp"

#include <cstdint>
#include <string>

namespace chess {

    // ============================================================
    // KNIGHTBOT NNUE V1
    // ============================================================

    constexpr int NNUE_V1_INPUT_COUNT =
        12 * 64;

    constexpr int NNUE_V1_HIDDEN_COUNT =
        128;

    constexpr std::uint32_t
        NNUE_V1_FORMAT_VERSION =
            1;


    // Keep the original names available so existing KnightBot
    // code continues to compile unchanged.
    constexpr int NNUE_INPUT_COUNT =
        NNUE_V1_INPUT_COUNT;

    constexpr int NNUE_HIDDEN_COUNT =
        NNUE_V1_HIDDEN_COUNT;

    constexpr std::uint32_t
        NNUE_FORMAT_VERSION =
            NNUE_V1_FORMAT_VERSION;


    // ============================================================
    // KNIGHTBOT HALFKP-512 / NNUE V2
    // ============================================================
    //
    // Feature transformer:
    //
    //      64 king squares
    //      *
    //      10 relative non-king piece classes
    //      *
    //      64 piece squares
    //
    //      = 40,960 features
    //
    // Two perspectives share the same feature transformer:
    //
    //      White king perspective -> 512
    //      Black king perspective -> 512
    //
    // Dense network:
    //
    //      512 + 512 + 13
    //            |
    //            v
    //           64
    //            |
    //            v
    //           32
    //            |
    //            v
    //            1
    //
    // Output is White-perspective centipawns.
    // ============================================================

    constexpr int HALFKP_FEATURE_COUNT =
        64 * 10 * 64;

    constexpr int HALFKP_TRANSFORMER_HIDDEN =
        512;

    constexpr int HALFKP_CONTEXT_COUNT =
        13;

    constexpr int HALFKP_DENSE1_COUNT =
        64;

    constexpr int HALFKP_DENSE2_COUNT =
        32;

    constexpr int HALFKP_DENSE1_INPUT_COUNT =
        HALFKP_TRANSFORMER_HIDDEN * 2 +
        HALFKP_CONTEXT_COUNT;

    constexpr std::uint32_t
        HALFKP_FORMAT_VERSION =
            2;


    // ============================================================
    // NETWORK MANAGEMENT
    // ============================================================

    bool loadNNUE(
        const std::string& path,
        std::string* errorMessage = nullptr
    );


    void unloadNNUE();


    bool nnueLoaded();


    std::string nnueLoadedFile();


    // Human-readable format:
    //
    //      "none"
    //      "NNUE v1"
    //      "HalfKP-512"
    std::string nnueFormatName();


    // Numeric format:
    //
    //      0 = no network
    //      1 = KNUE0001
    //      2 = KNUE0002
    int nnueFormatVersion();


    // ============================================================
    // ENABLE / DISABLE
    // ============================================================

    void setNNUEEnabled(
        bool enabled
    );


    bool nnueEnabled();


    // ============================================================
    // INFERENCE
    // ============================================================

    // White-perspective centipawn evaluation.
    //
    // Automatically dispatches to the loaded network format.
    //
    // Returns zero if no network is loaded.
    int evaluateNNUE(
        const Position& pos
    );
    // ============================================================
    // HALFKP INCREMENTAL ACCUMULATORS
    // ============================================================

    // Called after makeMove() has completely updated the board.
    void updateHalfKPAfterMove(
        Position& pos,
        const Move& move,
        const UndoState& undo
    );


    // Called after undoMove() has completely restored the board.
    void updateHalfKPAfterUndo(
        Position& pos,
        const Move& move,
        const UndoState& undo
    );


    // Forces HalfKP accumulators to be rebuilt from the board.
    //
    // Primarily useful for testing/debugging.
    void rebuildHalfKPAccumulators(
        Position& pos
    );


    // Full rebuild reference evaluator.
    //
    // This deliberately ignores the cached accumulators and is
    // used to prove that incremental evaluation is exact.
    int evaluateHalfKPFullRebuild(
        const Position& pos
    );

    // ============================================================
    // V1 FEATURE HELPER
    // ============================================================

    int nnueFeatureIndex(
        char piece,
        int square
    );


    // ============================================================
    // HALFKP-512 FEATURE HELPERS
    // ============================================================

    // perspectiveWhite:
    //
    //      true  = White king perspective
    //      false = Black king perspective
    //
    // Black perspective mirrors squares vertically using:
    //
    //      square ^ 56
    int halfKPFeatureIndex(
        bool perspectiveWhite,
        int kingSquare,
        char piece,
        int pieceSquare
    );


    // ============================================================
    // DEVELOPMENT TEST NETWORK
    // ============================================================
    //
    // Generates the original deterministic V1 material-only
    // compatibility network.
    //
    // This remains KNUE0001 and is not a HalfKP network.
    // ============================================================

    bool writeNNUECompatibilityTestNetwork(
        const std::string& path,
        std::string* errorMessage = nullptr
    );

} // namespace chess
