#pragma once

#include "chess.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace chess {

    constexpr int MATE_SCORE =
        30000;

    constexpr int MATE_THRESHOLD =
        29000;

    struct SearchResult {

        Move bestMove{};

        int score = 0;

        int depth = 0;

        std::uint64_t nodes = 0;

        double seconds = 0.0;

        bool hasMove = false;

        bool stopped = false;

        std::vector<Move> pv;
    };

    void clearTranspositionTable();

    std::size_t transpositionTableSize();

    SearchResult searchBestMove(
        const Position& pos,
        int maxDepth
    );

    SearchResult searchBestMoveTimed(
        const Position& pos,
        int milliseconds
    );

} // namespace chess