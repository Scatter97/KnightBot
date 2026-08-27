#pragma once

#include "chess.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace chess {

    constexpr int MATE_SCORE =
        30000;

    constexpr int MATE_THRESHOLD =
        29000;


    // ============================================================
    // SEARCH RESULT
    // ============================================================

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


    // ============================================================
    // SEARCH PROGRESS CALLBACK
    // ============================================================
    //
    // Called after every fully completed iterative-deepening depth.
    //
    // Example:
    //
    // depth 1
    // depth 2
    // depth 3
    // ...
    //
    // UCI mode uses this to print live "info depth ..." lines.
    //
    // ============================================================

    using SearchInfoCallback =
        std::function<
        void(
            const SearchResult&
            )
        >;


    // ============================================================
    // TRANSPOSITION TABLE
    // ============================================================

    void clearTranspositionTable();

    std::size_t transpositionTableSize();


    // ============================================================
    // FIXED DEPTH SEARCH
    // ============================================================

    SearchResult searchBestMove(
        const Position& pos,
        int maxDepth,
        const SearchInfoCallback& infoCallback = {}
    );


    // ============================================================
    // TIMED SEARCH
    // ============================================================

    SearchResult searchBestMoveTimed(
        const Position& pos,
        int milliseconds,
        const SearchInfoCallback& infoCallback = {}
    );

} // namespace chess