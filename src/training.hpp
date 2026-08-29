#pragma once

#include "chess.hpp"

#include <string>

namespace chess {

    // ============================================================
    // TRAINING DATA EXPORT
    // ============================================================
    //
    // Format:
    //
    //      FEN<TAB>targetCentipawns
    //
    // One position per line.
    //
    // The target may later come from:
    //
    //      Stockfish
    //      KnightBot self-play
    //      game outcome blending
    //      another teacher engine
    //
    // ============================================================

    bool appendTrainingPosition(
        const std::string& path,
        const Position& pos,
        int targetCentipawns,
        std::string* errorMessage = nullptr
    );

} // namespace chess
