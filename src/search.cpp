#include "search.hpp"

#include "evaluation.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>

namespace chess {

    namespace {

        constexpr int INF =
            32000;

        constexpr int MAX_PLY =
            128;

        constexpr int MAX_QPLY =
            8;

        constexpr int DELTA_MARGIN =
            150;


        // ============================================================
        // FIXED-SIZE TRANSPOSITION TABLE
        // ============================================================

        constexpr std::size_t TT_SIZE =
            1ULL << 20;

        constexpr std::size_t TT_MASK =
            TT_SIZE - 1;


        enum class TTFlag :
            std::uint8_t
        {
            Exact,
            LowerBound,
            UpperBound
        };


        struct TTEntry {

            std::uint64_t key = 0;

            int score = 0;

            int depth = -1;

            TTFlag flag =
                TTFlag::Exact;

            Move bestMove{};

            bool valid = false;
        };


        std::array<
            TTEntry,
            TT_SIZE
        > transpositionTable{};


        std::size_t ttUsed =
            0;


        TTEntry* probeTT(
            std::uint64_t key
        ) {
            TTEntry& entry =
                transpositionTable[
                    key &
                        TT_MASK
                ];

            if (
                entry.valid &&
                entry.key == key
                ) {
                return
                    &entry;
            }

            return nullptr;
        }


        void storeTT(
            std::uint64_t key,
            int depth,
            int score,
            TTFlag flag,
            const Move& move
        ) {
            TTEntry& entry =
                transpositionTable[
                    key &
                        TT_MASK
                ];


            if (
                !entry.valid
                ) {
                ++ttUsed;
            }


            if (
                !entry.valid ||
                entry.key != key ||
                depth >= entry.depth
                ) {
                entry.key =
                    key;

                entry.depth =
                    depth;

                entry.score =
                    score;

                entry.flag =
                    flag;

                entry.bestMove =
                    move;

                entry.valid =
                    true;
            }
        }


        // ============================================================
        // KILLERS / HISTORY
        // ============================================================

        std::array<
            std::array<Move, 2>,
            MAX_PLY
        > killerMoves{};


        std::array<
            std::array<
            std::array<int, 64>,
            64
            >,
            2
        > historyTable{};


        // ============================================================
        // SEARCH CONTEXT
        // ============================================================

        struct SearchContext {

            std::uint64_t nodes =
                0;

            bool useDeadline =
                false;

            std::chrono::steady_clock::time_point
                deadline{};

            bool stopped =
                false;
        };


        // ============================================================
        // HELPERS
        // ============================================================

        bool sameMove(
            const Move& a,
            const Move& b
        ) {
            return
                a.from == b.from &&
                a.to == b.to &&
                a.promotion == b.promotion &&
                a.enPassant == b.enPassant &&
                a.castle == b.castle;
        }


        bool validStoredMove(
            const Move& move
        ) {
            return
                move.from >= 0 &&
                move.from < 64 &&
                move.to >= 0 &&
                move.to < 64;
        }


        bool isCapture(
            const Position& pos,
            const Move& move
        ) {
            return
                move.enPassant ||
                pos.board[
                    move.to
                ] !=
                '.';
        }


        char capturedPieceForMove(
            const Position& pos,
            const Move& move
        ) {
            if (
                move.enPassant
                ) {
                return
                    pos.whiteToMove
                    ? 'p'
                    : 'P';
            }

            return
                pos.board[
                    move.to
                ];
        }


        // ============================================================
        // TIME
        // ============================================================

        bool timeExpired(
            SearchContext& context
        ) {
            if (
                !context.useDeadline
                ) {
                return false;
            }


            if (
                (
                    context.nodes &
                    4095ULL
                    )
                !=
                0
                ) {
                return false;
            }


            if (
                std::chrono::steady_clock::now()
                >=
                context.deadline
                ) {
                context.stopped =
                    true;

                return true;
            }


            return false;
        }


        // ============================================================
        // MOVE ORDERING
        // ============================================================

        int moveOrderScore(
            const Position& pos,
            const Move& move,
            const Move* ttMove,
            int ply
        ) {
            int score = 0;


            if (
                ttMove != nullptr &&
                sameMove(
                    move,
                    *ttMove
                )
                ) {
                score +=
                    1000000;
            }


            const char movingPiece =
                pos.board[
                    move.from
                ];


            const char capturedPiece =
                capturedPieceForMove(
                    pos,
                    move
                );


            if (
                move.promotion
                ) {
                score +=
                    30000;

                score +=
                    pieceValue(
                        move.promotion
                    );
            }


            if (
                capturedPiece != '.'
                ) {
                score +=
                    20000;

                score +=
                    pieceValue(
                        capturedPiece
                    )
                    *
                    10;

                score -=
                    pieceValue(
                        movingPiece
                    );
            }


            if (
                move.castle
                ) {
                score +=
                    1000;
            }


            if (
                ply >= 0 &&
                ply < MAX_PLY &&
                !isCapture(
                    pos,
                    move
                )
                ) {
                if (
                    validStoredMove(
                        killerMoves[
                            ply
                ][0]
                                )
                    &&
                    sameMove(
                        move,
                        killerMoves[
                            ply
                ][0]
                                )
                    ) {
                    score +=
                        4000;
                }

                else if (
                    validStoredMove(
                        killerMoves[
                            ply
                ][1]
                                )
                    &&
                    sameMove(
                        move,
                        killerMoves[
                            ply
                ][1]
                                )
                    ) {
                    score +=
                        3000;
                }


                const int side =
                    pos.whiteToMove
                    ? 0
                    : 1;

                score +=
                    historyTable[
                        side
                    ][
                        move.from
                    ][
                        move.to
                    ];
            }


            return score;
        }


        void orderMoves(
            const Position& pos,
            MoveList& moves,
            const Move* ttMove,
            int ply
        ) {
            std::array<
                int,
                MAX_MOVES
            > scores{};


            for (
                int i = 0;
                i < moves.count;
                ++i
                ) {
                scores[i] =
                    moveOrderScore(
                        pos,
                        moves[
                            static_cast<
                                std::size_t
                            >(i)
                        ],
                        ttMove,
                        ply
                    );
            }


            for (
                int i = 0;
                i <
                moves.count - 1;
                ++i
                ) {
                int bestIndex =
                    i;


                for (
                    int j = i + 1;
                    j < moves.count;
                    ++j
                    ) {
                    if (
                        scores[j] >
                        scores[bestIndex]
                        ) {
                        bestIndex =
                            j;
                    }
                }


                if (
                    bestIndex != i
                    ) {
                    std::swap(
                        scores[i],
                        scores[bestIndex]
                    );


                    std::swap(
                        moves[
                            static_cast<
                                std::size_t
                            >(i)
                        ],
                        moves[
                            static_cast<
                                std::size_t
                            >(bestIndex)
                        ]
                    );
                }
            }
        }


        void recordKiller(
            const Position& pos,
            const Move& move,
            int ply,
            int depth
        ) {
            if (
                ply < 0 ||
                ply >= MAX_PLY ||
                isCapture(
                    pos,
                    move
                )
                ) {
                return;
            }


            if (
                !sameMove(
                    move,
                    killerMoves[
                        ply
            ][0]
                            )
                ) {
                killerMoves[
                    ply
                ][1] =
                        killerMoves[
                            ply
                        ][0];

                    killerMoves[
                        ply
                    ][0] =
                            move;
            }


            const int side =
                pos.whiteToMove
                ? 0
                : 1;


            int& history =
                historyTable[
                    side
                ][
                    move.from
                ][
                    move.to
                ];


                    history +=
                        depth *
                        depth;


                    if (
                        history >
                        100000
                        ) {
                        history /=
                            2;
                    }
        }


        // ============================================================
        // BAD CAPTURE FILTER
        // ============================================================

        bool clearlyBadCapture(
            Position& pos,
            const Move& move
        ) {
            if (
                !isCapture(
                    pos,
                    move
                )
                ||
                move.promotion
                ) {
                return false;
            }


            const char attacker =
                pos.board[
                    move.from
                ];


            const char victim =
                capturedPieceForMove(
                    pos,
                    move
                );


            if (
                pieceValue(attacker)
                <=
                pieceValue(victim)
                +
                100
                ) {
                return false;
            }


            UndoState undo;


            makeMove(
                pos,
                move,
                undo
            );


            const bool attacked =
                isSquareAttacked(
                    pos,
                    move.to,
                    pos.whiteToMove
                );


            undoMove(
                pos,
                move,
                undo
            );


            return attacked;
        }


        // ============================================================
        // QUIESCENCE
        // ============================================================

        int quiescence(
            Position& pos,
            int alpha,
            int beta,
            int ply,
            int qply,
            SearchContext& context
        ) {
            ++context.nodes;


            if (
                timeExpired(
                    context
                )
                ) {
                return 0;
            }


            const bool checked =
                inCheck(
                    pos,
                    pos.whiteToMove
                );


            MoveList moves;


            generateLegalMoves(
                pos,
                moves
            );


            if (
                moves.empty()
                ) {
                if (
                    checked
                    ) {
                    return
                        -MATE_SCORE +
                        ply;
                }

                return 0;
            }


            if (
                qply >=
                MAX_QPLY
                ) {
                return
                    evaluateForSideToMove(
                        pos
                    );
            }


            // ========================================================
            // IN CHECK
            // ========================================================

            if (
                checked
                ) {
                orderMoves(
                    pos,
                    moves,
                    nullptr,
                    ply
                );


                for (
                    const Move& move :
                    moves
                    ) {
                    UndoState undo;


                    makeMove(
                        pos,
                        move,
                        undo
                    );


                    const int score =
                        -quiescence(
                            pos,
                            -beta,
                            -alpha,
                            ply + 1,
                            qply + 1,
                            context
                        );


                    undoMove(
                        pos,
                        move,
                        undo
                    );


                    if (
                        context.stopped
                        ) {
                        return 0;
                    }


                    if (
                        score >= beta
                        ) {
                        return score;
                    }


                    if (
                        score > alpha
                        ) {
                        alpha =
                            score;
                    }
                }


                return alpha;
            }


            // ========================================================
            // STAND PAT
            // ========================================================

            const int standPat =
                evaluateForSideToMove(
                    pos
                );


            if (
                standPat >= beta
                ) {
                return
                    standPat;
            }


            if (
                standPat > alpha
                ) {
                alpha =
                    standPat;
            }


            // ========================================================
            // TACTICAL MOVES
            // ========================================================

            MoveList tactical;


            for (
                const Move& move :
                moves
                ) {
                if (
                    !isCapture(
                        pos,
                        move
                    )
                    &&
                    !move.promotion
                    ) {
                    continue;
                }


                if (
                    isCapture(
                        pos,
                        move
                    )
                    &&
                    !move.promotion
                    ) {
                    const int gain =
                        pieceValue(
                            capturedPieceForMove(
                                pos,
                                move
                            )
                        );


                    if (
                        standPat +
                        gain +
                        DELTA_MARGIN
                        <
                        alpha
                        ) {
                        continue;
                    }
                }


                if (
                    clearlyBadCapture(
                        pos,
                        move
                    )
                    ) {
                    continue;
                }


                tactical.push_back(
                    move
                );
            }


            orderMoves(
                pos,
                tactical,
                nullptr,
                ply
            );


            for (
                const Move& move :
                tactical
                ) {
                UndoState undo;


                makeMove(
                    pos,
                    move,
                    undo
                );


                const int score =
                    -quiescence(
                        pos,
                        -beta,
                        -alpha,
                        ply + 1,
                        qply + 1,
                        context
                    );


                undoMove(
                    pos,
                    move,
                    undo
                );


                if (
                    context.stopped
                    ) {
                    return 0;
                }


                if (
                    score >= beta
                    ) {
                    return score;
                }


                if (
                    score > alpha
                    ) {
                    alpha =
                        score;
                }
            }


            return alpha;
        }


        // ============================================================
        // NEGAMAX
        // ============================================================

        int negamax(
            Position& pos,
            int depth,
            int alpha,
            int beta,
            int ply,
            SearchContext& context
        ) {
            if (
                ply >=
                MAX_PLY - 1
                ) {
                return
                    evaluateForSideToMove(
                        pos
                    );
            }


            if (
                depth <= 0
                ) {
                return
                    quiescence(
                        pos,
                        alpha,
                        beta,
                        ply,
                        0,
                        context
                    );
            }


            ++context.nodes;


            if (
                timeExpired(
                    context
                )
                ) {
                return 0;
            }


            const int originalAlpha =
                alpha;

            const int originalBeta =
                beta;


            const std::uint64_t key =
                pos.zobristKey;


            Move ttMove{};

            bool hasTTMove =
                false;


            TTEntry* ttEntry =
                probeTT(
                    key
                );


            if (
                ttEntry != nullptr
                ) {
                if (
                    validStoredMove(
                        ttEntry->bestMove
                    )
                    ) {
                    ttMove =
                        ttEntry->bestMove;

                    hasTTMove =
                        true;
                }


                if (
                    ttEntry->depth >=
                    depth
                    ) {
                    if (
                        ttEntry->flag ==
                        TTFlag::Exact
                        ) {
                        return
                            ttEntry->score;
                    }


                    if (
                        ttEntry->flag ==
                        TTFlag::LowerBound
                        ) {
                        alpha =
                            std::max(
                                alpha,
                                ttEntry->score
                            );
                    }

                    else if (
                        ttEntry->flag ==
                        TTFlag::UpperBound
                        ) {
                        beta =
                            std::min(
                                beta,
                                ttEntry->score
                            );
                    }


                    if (
                        alpha >= beta
                        ) {
                        return
                            ttEntry->score;
                    }
                }
            }


            MoveList moves;


            generateLegalMoves(
                pos,
                moves
            );


            if (
                moves.empty()
                ) {
                if (
                    inCheck(
                        pos,
                        pos.whiteToMove
                    )
                    ) {
                    return
                        -MATE_SCORE +
                        ply;
                }

                return 0;
            }


            orderMoves(
                pos,
                moves,
                hasTTMove
                ? &ttMove
                : nullptr,
                ply
            );


            int bestScore =
                -INF;


            Move bestMove =
                moves.front();


            for (
                const Move& move :
                moves
                ) {
                UndoState undo;


                makeMove(
                    pos,
                    move,
                    undo
                );


                const int score =
                    -negamax(
                        pos,
                        depth - 1,
                        -beta,
                        -alpha,
                        ply + 1,
                        context
                    );


                undoMove(
                    pos,
                    move,
                    undo
                );


                if (
                    context.stopped
                    ) {
                    return 0;
                }


                if (
                    score >
                    bestScore
                    ) {
                    bestScore =
                        score;

                    bestMove =
                        move;
                }


                if (
                    score >
                    alpha
                    ) {
                    alpha =
                        score;
                }


                if (
                    alpha >= beta
                    ) {
                    recordKiller(
                        pos,
                        move,
                        ply,
                        depth
                    );

                    break;
                }
            }


            TTFlag flag;


            if (
                bestScore <=
                originalAlpha
                ) {
                flag =
                    TTFlag::UpperBound;
            }

            else if (
                bestScore >=
                originalBeta
                ) {
                flag =
                    TTFlag::LowerBound;
            }

            else {
                flag =
                    TTFlag::Exact;
            }


            storeTT(
                key,
                depth,
                bestScore,
                flag,
                bestMove
            );


            return bestScore;
        }


        // ============================================================
        // PRINCIPAL VARIATION
        // ============================================================

        std::vector<Move> extractPV(
            Position pos,
            int maxDepth
        ) {
            std::vector<Move> pv;


            pv.reserve(
                static_cast<
                std::size_t
                >(
                    maxDepth
                    )
            );


            for (
                int i = 0;
                i < maxDepth;
                ++i
                ) {
                TTEntry* entry =
                    probeTT(
                        pos.zobristKey
                    );


                if (
                    entry == nullptr ||
                    !validStoredMove(
                        entry->bestMove
                    )
                    ) {
                    break;
                }


                const Move move =
                    entry->bestMove;


                MoveList legal;


                generateLegalMoves(
                    pos,
                    legal
                );


                bool found =
                    false;


                for (
                    const Move& candidate :
                    legal
                    ) {
                    if (
                        sameMove(
                            move,
                            candidate
                        )
                        ) {
                        found =
                            true;

                        break;
                    }
                }


                if (
                    !found
                    ) {
                    break;
                }


                pv.push_back(
                    move
                );


                makeMove(
                    pos,
                    move
                );
            }


            return pv;
        }


        // ============================================================
        // ROOT SEARCH
        // ============================================================

        SearchResult searchRoot(
            Position& pos,
            int depth,
            SearchContext& context
        ) {
            SearchResult result;


            result.depth =
                depth;


            MoveList moves;


            generateLegalMoves(
                pos,
                moves
            );


            if (
                moves.empty()
                ) {
                result.hasMove =
                    false;


                result.score =
                    inCheck(
                        pos,
                        pos.whiteToMove
                    )
                    ? -MATE_SCORE
                    : 0;


                return result;
            }


            result.hasMove =
                true;


            const std::uint64_t rootKey =
                pos.zobristKey;


            Move ttMove{};

            bool hasTTMove =
                false;


            if (
                TTEntry* entry =
                probeTT(
                    rootKey
                )
                ) {
                if (
                    validStoredMove(
                        entry->bestMove
                    )
                    ) {
                    ttMove =
                        entry->bestMove;

                    hasTTMove =
                        true;
                }
            }


            orderMoves(
                pos,
                moves,
                hasTTMove
                ? &ttMove
                : nullptr,
                0
            );


            int alpha =
                -INF;


            constexpr int beta =
                INF;


            int bestScore =
                -INF;


            Move bestMove =
                moves.front();


            for (
                const Move& move :
                moves
                ) {
                UndoState undo;


                makeMove(
                    pos,
                    move,
                    undo
                );


                const int score =
                    -negamax(
                        pos,
                        depth - 1,
                        -beta,
                        -alpha,
                        1,
                        context
                    );


                undoMove(
                    pos,
                    move,
                    undo
                );


                if (
                    context.stopped
                    ) {
                    break;
                }


                if (
                    score >
                    bestScore
                    ) {
                    bestScore =
                        score;

                    bestMove =
                        move;
                }


                if (
                    score >
                    alpha
                    ) {
                    alpha =
                        score;
                }
            }


            if (
                !context.stopped
                ) {
                result.bestMove =
                    bestMove;


                result.score =
                    bestScore;


                storeTT(
                    rootKey,
                    depth,
                    bestScore,
                    TTFlag::Exact,
                    bestMove
                );


                result.pv =
                    extractPV(
                        pos,
                        depth
                    );
            }


            return result;
        }


        // ============================================================
        // ITERATIVE DEEPENING
        // ============================================================

        SearchResult iterativeSearch(
            const Position& rootPosition,
            int maxDepth,
            bool useDeadline,
            std::chrono::steady_clock::time_point deadline
        ) {
            const auto start =
                std::chrono::steady_clock::now();


            Position pos =
                rootPosition;


            SearchContext context;


            context.useDeadline =
                useDeadline;


            context.deadline =
                deadline;


            SearchResult bestCompleted;


            MoveList rootMoves;


            generateLegalMoves(
                pos,
                rootMoves
            );


            if (
                rootMoves.empty()
                ) {
                bestCompleted.hasMove =
                    false;


                bestCompleted.score =
                    inCheck(
                        pos,
                        pos.whiteToMove
                    )
                    ? -MATE_SCORE
                    : 0;


                const auto end =
                    std::chrono::steady_clock::now();


                bestCompleted.seconds =
                    std::chrono::duration<double>(
                        end - start
                    ).count();


                return bestCompleted;
            }


            bestCompleted.hasMove =
                true;


            bestCompleted.bestMove =
                rootMoves.front();


            for (
                int depth = 1;
                depth <= maxDepth;
                ++depth
                ) {
                if (
                    useDeadline &&
                    std::chrono::steady_clock::now()
                    >=
                    deadline
                    ) {
                    break;
                }


                SearchResult current =
                    searchRoot(
                        pos,
                        depth,
                        context
                    );


                if (
                    context.stopped
                    ) {
                    break;
                }


                current.nodes =
                    context.nodes;


                current.depth =
                    depth;


                bestCompleted =
                    current;


                if (
                    std::abs(
                        current.score
                    )
                    >=
                    MATE_THRESHOLD
                    ) {
                    break;
                }
            }


            const auto end =
                std::chrono::steady_clock::now();


            bestCompleted.nodes =
                context.nodes;


            bestCompleted.seconds =
                std::chrono::duration<double>(
                    end - start
                ).count();


            bestCompleted.stopped =
                context.stopped;


            return bestCompleted;
        }

    } // anonymous namespace



    // ============================================================
    // PUBLIC API
    // ============================================================

    void clearTranspositionTable() {

        for (
            TTEntry& entry :
            transpositionTable
            ) {
            entry =
                TTEntry{};
        }


        ttUsed =
            0;


        killerMoves =
        {};


        historyTable =
        {};
    }


    std::size_t transpositionTableSize() {

        return
            ttUsed;
    }


    SearchResult searchBestMove(
        const Position& pos,
        int maxDepth
    ) {
        if (
            maxDepth < 1
            ) {
            maxDepth =
                1;
        }


        return
            iterativeSearch(
                pos,
                maxDepth,
                false,
                {}
            );
    }


    SearchResult searchBestMoveTimed(
        const Position& pos,
        int milliseconds
    ) {
        if (
            milliseconds < 1
            ) {
            milliseconds =
                1;
        }


        const auto deadline =
            std::chrono::steady_clock::now()
            +
            std::chrono::milliseconds(
                milliseconds
            );


        return
            iterativeSearch(
                pos,
                64,
                true,
                deadline
            );
    }

} // namespace chess