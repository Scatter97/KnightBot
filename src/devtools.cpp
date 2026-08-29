#include "devtools.hpp"

#include "chess.hpp"
#include "evaluation.hpp"
#include "nnue.hpp"
#include "search.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace chess {

    namespace {

        bool sameMove(
            const Move& a,
            const Move& b
        ) {
            return
                a.from ==
                b.from
                &&
                a.to ==
                b.to
                &&
                a.promotion ==
                b.promotion
                &&
                a.enPassant ==
                b.enPassant
                &&
                a.castle ==
                b.castle;
        }


        bool playUciMoveForTest(
            Position& pos,
            const std::string& uci
        ) {
            MoveList moves;


            generateLegalMoves(
                pos,
                moves
            );


            for (
                const Move& move :
                moves
                ) {
                if (
                    moveToUci(
                        move
                    )
                    ==
                    uci
                    ) {
                    makeMove(
                        pos,
                        move
                    );


                    return true;
                }
            }


            return false;
        }


        void printTestResult(
            const std::string& name,
            bool passed
        ) {
            std::cout
                <<
                std::left
                <<
                std::setw(38)
                <<
                name
                <<
                (
                    passed
                    ? "PASS"
                    : "FAIL"
                )
                <<
                '\n';
        }


        std::uint64_t checksumMix(
            std::uint64_t hash,
            std::uint64_t value
        ) {
            hash ^=
                value;


            hash *=
                1099511628211ULL;


            return hash;
        }

        void addSearchStats(
            SearchStats& total,
            const SearchStats& value
        ) {
            total.qsearchNodes += value.qsearchNodes;

            total.evaluations += value.evaluations;

            total.evalCacheHits += value.evalCacheHits;
            total.evalCacheMisses += value.evalCacheMisses;

            total.ttProbes += value.ttProbes;
            total.ttHits += value.ttHits;
            total.ttCutoffs += value.ttCutoffs;

            total.betaCutoffs += value.betaCutoffs;

            total.nullMoveAttempts += value.nullMoveAttempts;
            total.nullMoveCutoffs += value.nullMoveCutoffs;

            total.lmrReductions += value.lmrReductions;
            total.lmrResearches += value.lmrResearches;

            total.pvsResearches += value.pvsResearches;

            total.aspirationResearches += value.aspirationResearches;

            total.razorAttempts += value.razorAttempts;
            total.razorCutoffs += value.razorCutoffs;

            total.reverseFutilityCutoffs +=
                value.reverseFutilityCutoffs;

            total.moveFutilityPrunes +=
                value.moveFutilityPrunes;

            total.checkExtensions += value.checkExtensions;

            total.qsearchDeltaPrunes +=
                value.qsearchDeltaPrunes;

            total.qsearchSeePrunes +=
                value.qsearchSeePrunes;

            total.halfKPPendingChildren +=
                value.halfKPPendingChildren;

            total.halfKPMaterializations +=
                value.halfKPMaterializations;

            total.halfKPLazySkips +=
                value.halfKPLazySkips;
        }

    } // anonymous namespace


    // ============================================================
    // SELF TEST
    // ============================================================

    bool runSelfTest() {
        std::cout
            <<
            "\nKnightBot self-test\n"
            "========================================\n";


        bool allPassed =
            true;


        // ========================================================
        // STARTPOS PERFT
        // ========================================================

        {
            const Position pos =
                startPosition();


            const bool passed =
                perft(
                    pos,
                    4
                )
                ==
                197281ULL;


            printTestResult(
                "Startpos perft depth 4",
                passed
            );


            allPassed &=
                passed;
        }


        // ========================================================
        // KIWIPETE
        // ========================================================

        {
            const Position pos =
                fromFEN(
                    "r3k2r/p1ppqpb1/bn2pnp1/"
                    "3PN3/1p2P3/2N2Q1p/"
                    "PPPBBPPP/R3K2R "
                    "w KQkq - 0 1"
                );


            const bool passed =
                perft(
                    pos,
                    3
                )
                ==
                97862ULL;


            printTestResult(
                "Kiwipete perft depth 3",
                passed
            );


            allPassed &=
                passed;
        }


        // ========================================================
        // MATE IN ONE
        // ========================================================

        {
            const Position pos =
                fromFEN(
                    "7k/5Q2/6K1/8/8/8/8/8 "
                    "w - - 0 1"
                );


            clearTranspositionTable();

            resetSearchStop();


            const SearchResult result =
                searchBestMove(
                    pos,
                    3
                );


            const bool passed =
                result.hasMove
                &&
                result.score >=
                MATE_THRESHOLD;


            printTestResult(
                "Mate-in-one search",
                passed
            );


            allPassed &=
                passed;
        }


        // ========================================================
        // FIFTY-MOVE RULE
        // ========================================================

        {
            const Position pos =
                fromFEN(
                    "8/8/8/8/8/8/R7/K6k "
                    "w - - 100 51"
                );


            SearchHistory history;

            history.push_back(
                pos.zobristKey
            );


            clearTranspositionTable();

            resetSearchStop();


            const SearchResult result =
                searchBestMove(
                    pos,
                    4,
                    {},
                    history
                );


            const bool passed =
                result.score ==
                0;


            printTestResult(
                "50-move rule",
                passed
            );


            allPassed &=
                passed;
        }


        // ========================================================
        // THREEFOLD REPETITION
        // ========================================================

        {
            Position pos =
                startPosition();


            SearchHistory history;

            history.push_back(
                pos.zobristKey
            );


            const std::vector<std::string>
                moves{
                    "g1f3",
                    "g8f6",
                    "f3g1",
                    "f6g8",
                    "g1f3",
                    "g8f6",
                    "f3g1",
                    "f6g8"
                };


            bool movesValid =
                true;


            for (
                const std::string& move :
                moves
                ) {
                if (
                    !playUciMoveForTest(
                        pos,
                        move
                    )
                    ) {
                    movesValid =
                        false;

                    break;
                }


                history.push_back(
                    pos.zobristKey
                );
            }


            clearTranspositionTable();

            resetSearchStop();


            const SearchResult result =
                searchBestMove(
                    pos,
                    4,
                    {},
                    history
                );


            const bool passed =
                movesValid
                &&
                result.score ==
                0;


            printTestResult(
                "Threefold repetition",
                passed
            );


            allPassed &=
                passed;
        }


        // ========================================================
        // ZOBRIST MAKE/UNDO
        // ========================================================

        {
            Position pos =
                startPosition();


            const std::uint64_t
                originalKey =
                    pos.zobristKey;


            MoveList moves;

            generateLegalMoves(
                pos,
                moves
            );


            bool passed =
                !moves.empty();


            if (
                passed
                ) {
                const Move move =
                    moves.front();


                UndoState undo;


                makeMove(
                    pos,
                    move,
                    undo
                );


                undoMove(
                    pos,
                    move,
                    undo
                );


                passed =
                    pos.zobristKey ==
                    originalKey;
            }


            printTestResult(
                "Zobrist make/undo",
                passed
            );


            allPassed &=
                passed;
        }


        // ========================================================
        // NNUE FEATURE MAP
        // ========================================================

        {
            bool passed =
                true;


            passed &=
                nnueFeatureIndex(
                    'P',
                    0
                )
                ==
                WP * 64;


            passed &=
                nnueFeatureIndex(
                    'k',
                    63
                )
                ==
                BK * 64 +
                63;


            passed &=
                nnueFeatureIndex(
                    '.',
                    0
                )
                ==
                -1;


            printTestResult(
                "NNUE feature mapping",
                passed
            );


            allPassed &=
                passed;
        }


        std::cout
            <<
            "========================================\n"
            <<
            (
                allPassed
                ? "SELF-TEST PASSED\n"
                : "SELF-TEST FAILED\n"
            );


        return
            allPassed;
    }


    // ============================================================
    // BENCHMARK
    // ============================================================

    void runBenchmark() {
        struct BenchPosition {

            const char* fen;

            int depth;
        };


        const std::vector<BenchPosition>
            positions{
                {
                    "rnbqkbnr/pppppppp/8/8/8/8/"
                    "PPPPPPPP/RNBQKBNR "
                    "w KQkq - 0 1",
                    6
                },

                {
                    "r3k2r/p1ppqpb1/bn2pnp1/"
                    "3PN3/1p2P3/2N2Q1p/"
                    "PPPBBPPP/R3K2R "
                    "w KQkq - 0 1",
                    5
                },

                {
                    "r1bq1rk1/pp2bppp/2n1pn2/"
                    "2pp4/3P4/2PBPN2/"
                    "PP1N1PPP/R1BQ1RK1 "
                    "w - - 3 9",
                    6
                },

                {
                    "8/2p5/3p4/1P1P4/"
                    "2P1k3/4P3/5K2/8 "
                    "w - - 0 40",
                    7
                },

                {
                    "4rrk1/1pp2ppp/p1n5/"
                    "3q4/3P4/P1P1Q3/"
                    "1P3PPP/2RR2K1 "
                    "w - - 0 22",
                    6
                },

                {
                    "2r3k1/5pp1/4p2p/"
                    "3pP3/3P1P2/6P1/"
                    "5K1P/2R5 "
                    "w - - 0 32",
                    7
                }
            };


        std::uint64_t totalNodes =
            0;


        std::uint64_t checksum =
            1469598103934665603ULL;


        const auto start =
            std::chrono::steady_clock::now();


        for (
            std::size_t index = 0;
            index <
            positions.size();
            ++index
            ) {
            const Position pos =
                fromFEN(
                    positions[
                        index
                    ].fen
                );


            SearchHistory history;

            history.push_back(
                pos.zobristKey
            );


            clearTranspositionTable();

            resetSearchStop();


            const SearchResult result =
                searchBestMove(
                    pos,
                    positions[
                        index
                    ].depth,
                    {},
                    history
                );


            totalNodes +=
                result.nodes;


            checksum =
                checksumMix(
                    checksum,
                    static_cast<
                        std::uint64_t
                    >(
                        result.score +
                        32000
                    )
                );


            checksum =
                checksumMix(
                    checksum,
                    static_cast<
                        std::uint64_t
                    >(
                        result.bestMove.from +
                        1
                    )
                );


            checksum =
                checksumMix(
                    checksum,
                    static_cast<
                        std::uint64_t
                    >(
                        result.bestMove.to +
                        1
                    )
                );


            std::cout
                <<
                "bench "
                <<
                (
                    index +
                    1
                )
                <<
                "/"
                <<
                positions.size()
                <<
                " depth "
                <<
                result.depth
                <<
                " score "
                <<
                result.score
                <<
                " nodes "
                <<
                result.nodes
                <<
                " bestmove "
                <<
                (
                    result.hasMove
                    ?
                    moveToUci(
                        result.bestMove
                    )
                    :
                    "0000"
                )
                <<
                '\n';
        }


        const auto end =
            std::chrono::steady_clock::now();


        const double seconds =
            std::chrono::duration<double>(
                end -
                start
            ).count();


        const std::uint64_t nps =
            seconds > 0.0
            ?
            static_cast<std::uint64_t>(
                static_cast<double>(
                    totalNodes
                )
                /
                seconds
            )
            :
            0;


        std::cout
            <<
            "\nKnightBot benchmark\n"
            <<
            "Positions: "
            <<
            positions.size()
            <<
            '\n'
            <<
            "Nodes:     "
            <<
            totalNodes
            <<
            '\n'
            <<
            std::fixed
            <<
            std::setprecision(3)
            <<
            "Time:      "
            <<
            seconds
            <<
            " s\n"
            <<
            "NPS:       "
            <<
            nps
            <<
            '\n'
            <<
            "Checksum:  "
            <<
            checksum
            <<
            '\n'
            <<
            "Evaluator: "
            <<
            (
                nnueEnabled() &&
                nnueLoaded()
                ?
                "NNUE"
                :
                "Handcrafted"
            )
            <<
            '\n';
    }

    void runFullBenchmark() {
        struct BenchPosition {
            const char* name;
            const char* category;
            const char* fen;
            int depth;
        };


        const std::vector<BenchPosition> positions{
            {
                "Start position",
                "Opening",
                "rnbqkbnr/pppppppp/8/8/8/8/"
                "PPPPPPPP/RNBQKBNR "
                "w KQkq - 0 1",
                6
            },

            {
                "Kiwipete",
                "Castling / tactical",
                "r3k2r/p1ppqpb1/bn2pnp1/"
                "3PN3/1p2P3/2N2Q1p/"
                "PPPBBPPP/R3K2R "
                "w KQkq - 0 1",
                5
            },

            {
                "Closed middlegame",
                "Middlegame",
                "r1bq1rk1/pp2bppp/2n1pn2/"
                "2pp4/3P4/2PBPN2/"
                "PP1N1PPP/R1BQ1RK1 "
                "w - - 3 9",
                6
            },

            {
                "King and pawn ending",
                "Pawn ending",
                "8/2p5/3p4/1P1P4/"
                "2P1k3/4P3/5K2/8 "
                "w - - 0 40",
                7
            },

            {
                "Heavy-piece middlegame",
                "Tactical middlegame",
                "4rrk1/1pp2ppp/p1n5/"
                "3q4/3P4/P1P1Q3/"
                "1P3PPP/2RR2K1 "
                "w - - 0 22",
                6
            },

            {
                "Rook ending",
                "Endgame",
                "2r3k1/5pp1/4p2p/"
                "3pP3/3P1P2/6P1/"
                "5K1P/2R5 "
                "w - - 0 32",
                7
            },

            {
                "Rook and pawn tactics",
                "Endgame tactics",
                "8/2p5/3p4/KP5r/"
                "1R3p1k/8/4P1P1/8 "
                "w - - 0 1",
                7
            },

            {
                "Castling stress",
                "Castling / attack",
                "r3k2r/Pppp1ppp/1b3nbN/"
                "nP6/BBP1P3/q4N2/"
                "Pp1P2PP/R2Q1RK1 "
                "w kq - 0 1",
                6
            },

            {
                "Development tactics",
                "Tactical",
                "rnbq1k1r/pp1Pbppp/2p2n2/"
                "8/2B5/8/"
                "PPP1NPPP/RNBQK2R "
                "w KQ - 1 8",
                6
            },

            {
                "Open middlegame",
                "Middlegame",
                "r4rk1/1pp1qppp/p1np1n2/"
                "8/2B1P3/2N2Q1P/"
                "PPP2PP1/R1B2RK1 "
                "w - - 0 10",
                6
            },

            {
                "Mate pressure",
                "King attack",
                "7k/5Q2/6K1/8/8/8/8/8 "
                "w - - 0 1",
                5
            },

            {
                "Black to move opening",
                "Opening",
                "rnbqkbnr/pppp1ppp/8/"
                "4p3/4P3/8/"
                "PPPP1PPP/RNBQKBNR "
                "b KQkq - 0 1",
                6
            }
        };


        SearchStats totalStats{};

        std::uint64_t totalNodes =
            0;

        std::uint64_t checksum =
            1469598103934665603ULL;

        std::vector<std::uint64_t> positionNps;

        positionNps.reserve(
            positions.size()
        );


        std::cout
            <<
            "\nKnightBot Full Benchmark\n"
            "============================================================\n"
            <<
            "Evaluator: "
            <<
            (
                nnueEnabled() &&
                nnueLoaded()
                ?
                "NNUE"
                :
                "Handcrafted"
            )
            <<
            "\nPositions: "
            <<
            positions.size()
            <<
            "\n============================================================\n";


        const auto totalStart =
            std::chrono::steady_clock::now();


        for (
            std::size_t index = 0;
            index < positions.size();
            ++index
            ) {
            const BenchPosition& bench =
                positions[index];

            const Position pos =
                fromFEN(
                    bench.fen
                );


            SearchHistory history;

            history.push_back(
                pos.zobristKey
            );


            clearTranspositionTable();

            resetSearchStop();


            SearchStats stats{};

            const SearchResult result =
                searchBestMoveProfiled(
                    pos,
                    bench.depth,
                    stats,
                    {},
                    history
                );


            totalNodes +=
                result.nodes;

            addSearchStats(
                totalStats,
                stats
            );


            checksum =
                checksumMix(
                    checksum,
                    static_cast<std::uint64_t>(
                        result.score +
                        32000
                    )
                );

            checksum =
                checksumMix(
                    checksum,
                    static_cast<std::uint64_t>(
                        result.bestMove.from +
                        1
                    )
                );

            checksum =
                checksumMix(
                    checksum,
                    static_cast<std::uint64_t>(
                        result.bestMove.to +
                        1
                    )
                );


            const std::uint64_t nps =
                result.seconds > 0.0
                ?
                static_cast<std::uint64_t>(
                    static_cast<double>(
                        result.nodes
                    )
                    /
                    result.seconds
                )
                :
                0;


            positionNps.push_back(
                nps
            );


            std::cout
                <<
                "\n["
                <<
                (
                    index +
                    1
                )
                <<
                "/"
                <<
                positions.size()
                <<
                "] "
                <<
                bench.name
                <<
                "\n"
                <<
                "Category:       "
                <<
                bench.category
                <<
                "\n"
                <<
                "Depth:          "
                <<
                result.depth
                <<
                "\n"
                <<
                "Score:          "
                <<
                result.score
                <<
                " cp\n"
                <<
                "Best move:      "
                <<
                (
                    result.hasMove
                    ?
                    moveToUci(
                        result.bestMove
                    )
                    :
                    "0000"
                )
                <<
                "\n"
                <<
                "Nodes:          "
                <<
                result.nodes
                <<
                "\n"
                <<
                "QSearch nodes:  "
                <<
                stats.qsearchNodes
                <<
                "\n"
                <<
                "Time:           "
                <<
                std::fixed
                <<
                std::setprecision(3)
                <<
                result.seconds
                <<
                " s\n"
                <<
                "NPS:            "
                <<
                nps
                <<
                "\n";
        }


        const auto totalEnd =
            std::chrono::steady_clock::now();

        const double totalSeconds =
            std::chrono::duration<double>(
                totalEnd -
                totalStart
            ).count();


        const std::uint64_t overallNps =
            totalSeconds > 0.0
            ?
            static_cast<std::uint64_t>(
                static_cast<double>(
                    totalNodes
                )
                /
                totalSeconds
            )
            :
            0;


        std::sort(
            positionNps.begin(),
            positionNps.end()
        );


        const std::uint64_t minNps =
            positionNps.empty()
            ?
            0
            :
            positionNps.front();

        const std::uint64_t maxNps =
            positionNps.empty()
            ?
            0
            :
            positionNps.back();


        std::uint64_t medianNps =
            0;

        if (
            !positionNps.empty()
            ) {
            const std::size_t middle =
                positionNps.size() /
                2;

            if (
                positionNps.size() %
                2 ==
                0
                ) {
                medianNps =
                    (
                        positionNps[middle - 1] +
                        positionNps[middle]
                    )
                    /
                    2;
            }

            else {
                medianNps =
                    positionNps[middle];
            }
        }


        const double qsearchNodeRate =
            totalNodes >
            0
            ?
            100.0 *
            static_cast<double>(
                totalStats.qsearchNodes
            )
            /
            static_cast<double>(
                totalNodes
            )
            :
            0.0;


        const double evalCacheHitRate =
            (
                totalStats.evalCacheHits +
                totalStats.evalCacheMisses
            ) >
            0
            ?
            100.0 *
            static_cast<double>(
                totalStats.evalCacheHits
            )
            /
            static_cast<double>(
                totalStats.evalCacheHits +
                totalStats.evalCacheMisses
            )
            :
            0.0;


        const double ttHitRate =
            totalStats.ttProbes >
            0
            ?
            100.0 *
            static_cast<double>(
                totalStats.ttHits
            )
            /
            static_cast<double>(
                totalStats.ttProbes
            )
            :
            0.0;


        const double nullCutoffRate =
            totalStats.nullMoveAttempts >
            0
            ?
            100.0 *
            static_cast<double>(
                totalStats.nullMoveCutoffs
            )
            /
            static_cast<double>(
                totalStats.nullMoveAttempts
            )
            :
            0.0;


        const double lazySaveRate =
            totalStats.halfKPPendingChildren >
            0
            ?
            100.0 *
            static_cast<double>(
                totalStats.halfKPLazySkips
            )
            /
            static_cast<double>(
                totalStats.halfKPPendingChildren
            )
            :
            0.0;


        std::cout
            <<
            "\n============================================================\n"
            "FULL BENCHMARK SUMMARY\n"
            "============================================================\n"
            "\nPERFORMANCE\n"
            "------------------------------------------------------------\n"
            "Nodes:                    "
            <<
            totalNodes
            <<
            "\n"
            "Time:                     "
            <<
            std::fixed
            <<
            std::setprecision(3)
            <<
            totalSeconds
            <<
            " s\n"
            "Overall NPS:              "
            <<
            overallNps
            <<
            "\n"
            "Minimum position NPS:     "
            <<
            minNps
            <<
            "\n"
            "Median position NPS:      "
            <<
            medianNps
            <<
            "\n"
            "Maximum position NPS:     "
            <<
            maxNps
            <<
            "\n"
            "\nSEARCH\n"
            "------------------------------------------------------------\n"
            "QSearch nodes:            "
            <<
            totalStats.qsearchNodes
            <<
            "\n"
            "QSearch node share:       "
            <<
            std::setprecision(2)
            <<
            qsearchNodeRate
            <<
            "%\n"
            "QSearch delta prunes:     "
            <<
            totalStats.qsearchDeltaPrunes
            <<
            "\n"
            "QSearch SEE prunes:       "
            <<
            totalStats.qsearchSeePrunes
            <<
            "\n"
            "Beta cutoffs:             "
            <<
            totalStats.betaCutoffs
            <<
            "\n"
            "Razor attempts:           "
            <<
            totalStats.razorAttempts
            <<
            "\n"
            "Razor cutoffs:            "
            <<
            totalStats.razorCutoffs
            <<
            "\n"
            "Reverse futility cutoffs: "
            <<
            totalStats.reverseFutilityCutoffs
            <<
            "\n"
            "Move futility prunes:     "
            <<
            totalStats.moveFutilityPrunes
            <<
            "\n"
            "Check extensions:         "
            <<
            totalStats.checkExtensions
            <<
            "\n"
            "LMR reductions:           "
            <<
            totalStats.lmrReductions
            <<
            "\n"
            "LMR re-searches:          "
            <<
            totalStats.lmrResearches
            <<
            "\n"
            "PVS re-searches:          "
            <<
            totalStats.pvsResearches
            <<
            "\n"
            "Aspiration re-searches:   "
            <<
            totalStats.aspirationResearches
            <<
            "\n"
            "\nTRANSPOSITION TABLE\n"
            "------------------------------------------------------------\n"
            "TT probes:                "
            <<
            totalStats.ttProbes
            <<
            "\n"
            "TT hits:                  "
            <<
            totalStats.ttHits
            <<
            "\n"
            "TT hit rate:              "
            <<
            std::setprecision(2)
            <<
            ttHitRate
            <<
            "%\n"
            "TT cutoffs:               "
            <<
            totalStats.ttCutoffs
            <<
            "\n"
            "\nNULL MOVE\n"
            "------------------------------------------------------------\n"
            "Attempts:                 "
            <<
            totalStats.nullMoveAttempts
            <<
            "\n"
            "Cutoffs:                  "
            <<
            totalStats.nullMoveCutoffs
            <<
            "\n"
            "Cutoff rate:              "
            <<
            nullCutoffRate
            <<
            "%\n"
            "\nEVALUATION\n"
            "------------------------------------------------------------\n"
            "Actual evaluations:       "
            <<
            totalStats.evaluations
            <<
            "\n"
            "Eval-cache hits:          "
            <<
            totalStats.evalCacheHits
            <<
            "\n"
            "Eval-cache misses:        "
            <<
            totalStats.evalCacheMisses
            <<
            "\n"
            "Eval-cache hit rate:      "
            <<
            evalCacheHitRate
            <<
            "%\n"
            "\nHALFKP LAZY ACCUMULATORS\n"
            "------------------------------------------------------------\n"
            "Pending children:         "
            <<
            totalStats.halfKPPendingChildren
            <<
            "\n"
            "Materializations:         "
            <<
            totalStats.halfKPMaterializations
            <<
            "\n"
            "Lazy skips:               "
            <<
            totalStats.halfKPLazySkips
            <<
            "\n"
            "Lazy save rate:           "
            <<
            lazySaveRate
            <<
            "%\n"
            "\nREGRESSION\n"
            "------------------------------------------------------------\n"
            "Checksum:                 "
            <<
            checksum
            <<
            "\n"
            "Evaluator:                "
            <<
            (
                nnueEnabled() &&
                nnueLoaded()
                ?
                "NNUE"
                :
                "Handcrafted"
            )
            <<
            "\n"
            "============================================================\n";
    }

} // namespace chess
