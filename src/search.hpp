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

    struct SearchStats {
        std::uint64_t qsearchNodes = 0;

        std::uint64_t evaluations = 0;

        std::uint64_t evalCacheHits = 0;
        std::uint64_t evalCacheMisses = 0;

        std::uint64_t ttProbes = 0;
        std::uint64_t ttHits = 0;
        std::uint64_t ttCutoffs = 0;

        std::uint64_t betaCutoffs = 0;

        std::uint64_t nullMoveAttempts = 0;
        std::uint64_t nullMoveCutoffs = 0;

        std::uint64_t lmrReductions = 0;
        std::uint64_t lmrResearches = 0;

        std::uint64_t pvsResearches = 0;

        std::uint64_t aspirationResearches = 0;

        std::uint64_t razorAttempts = 0;
        std::uint64_t razorCutoffs = 0;

        std::uint64_t reverseFutilityCutoffs = 0;
        std::uint64_t moveFutilityPrunes = 0;

        std::uint64_t checkExtensions = 0;

        std::uint64_t qsearchDeltaPrunes = 0;
        std::uint64_t qsearchSeePrunes = 0;

        std::uint64_t halfKPPendingChildren = 0;
        std::uint64_t halfKPMaterializations = 0;
        std::uint64_t halfKPLazySkips = 0;
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

    using SearchHistory =
        std::vector<std::uint64_t>;


    // May be called from the UCI input thread while a search is active.
    void requestSearchStop();

    void resetSearchStop();


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
        const SearchInfoCallback& infoCallback = {},
        const SearchHistory& history = {}
    );

    SearchResult searchBestMoveProfiled(
        const Position& pos,
        int maxDepth,
        SearchStats& stats,
        const SearchInfoCallback& infoCallback = {},
        const SearchHistory& history = {}
    );


    // ============================================================
    // TIMED SEARCH
    // ============================================================

    SearchResult searchBestMoveTimed(
        const Position& pos,
        int milliseconds,
        const SearchInfoCallback& infoCallback = {},
        const SearchHistory& history = {}
    );

    SearchResult searchBestMoveTimed(
        const Position& pos,
        int softMilliseconds,
        int hardMilliseconds,
        const SearchInfoCallback& infoCallback,
        const SearchHistory& history = {}
    );

} // namespace chess
