#include "search.hpp"

#include "evaluation.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <vector>

namespace chess {

    namespace {

        constexpr int INF = 32000;
        constexpr int MAX_PLY = 128;
        constexpr int MAX_QPLY = 8;
        constexpr int DELTA_MARGIN = 150;


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


        std::size_t ttUsed = 0;


        // ============================================================
        // TT PROBE
        // ============================================================

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
                return &entry;
            }

            return nullptr;
        }


        // ============================================================
        // TT STORE
        // ============================================================

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
        // KILLER MOVES
        // ============================================================

        std::array<
            std::array<Move, 2>,
            MAX_PLY
        > killerMoves{};


        // ============================================================
        // HISTORY HEURISTIC
        // ============================================================

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

            std::uint64_t nodes = 0;

            bool useDeadline = false;

            std::chrono::steady_clock::time_point
                deadline{};

            bool stopped = false;
        };


        // ============================================================
        // BASIC HELPERS
        // ============================================================

        int fileOf(
            int square
        ) {
            return
                square &
                7;
        }


        int rankOf(
            int square
        ) {
            return
                square >>
                3;
        }


        bool onBoard(
            int file,
            int rank
        ) {
            return
                file >= 0 &&
                file < 8 &&
                rank >= 0 &&
                rank < 8;
        }


        bool isWhitePiece(
            char piece
        ) {
            return
                piece >= 'A' &&
                piece <= 'Z';
        }


        bool isBlackPiece(
            char piece
        ) {
            return
                piece >= 'a' &&
                piece <= 'z';
        }


        bool pieceBelongsToSide(
            char piece,
            bool white
        ) {
            if (
                white
                ) {
                return
                    isWhitePiece(
                        piece
                    );
            }

            return
                isBlackPiece(
                    piece
                );
        }


        // ============================================================
        // MOVE COMPARISON
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


        // ============================================================
        // STORED MOVE VALIDITY
        // ============================================================

        bool validStoredMove(
            const Move& move
        ) {
            return
                move.from >= 0 &&
                move.from < 64 &&
                move.to >= 0 &&
                move.to < 64;
        }


        // ============================================================
        // CAPTURE DETECTION
        // ============================================================

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


        // ============================================================
        // CAPTURED PIECE
        // ============================================================

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
        // TIME CHECK
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
        // STATIC EXCHANGE EVALUATION
        // ============================================================

        using SeeBoard =
            std::array<
            char,
            64
            >;


        // ============================================================
        // SEE PIECE ATTACK
        // ============================================================

        bool seePieceAttacksSquare(
            const SeeBoard& board,
            int from,
            int target
        ) {
            const char piece =
                board[
                    from
                ];

            if (
                piece == '.'
                ) {
                return false;
            }

            const int fromFile =
                fileOf(
                    from
                );

            const int fromRank =
                rankOf(
                    from
                );

            const int targetFile =
                fileOf(
                    target
                );

            const int targetRank =
                rankOf(
                    target
                );

            const int df =
                targetFile -
                fromFile;

            const int dr =
                targetRank -
                fromRank;

            const char type =
                static_cast<char>(
                    std::tolower(
                        static_cast<unsigned char>(
                            piece
                            )
                    )
                    );


            // ========================================================
            // PAWN
            // ========================================================

            if (
                type == 'p'
                ) {
                if (
                    isWhitePiece(
                        piece
                    )
                    ) {
                    return
                        dr == 1 &&
                        std::abs(
                            df
                        )
                        ==
                        1;
                }

                return
                    dr == -1 &&
                    std::abs(
                        df
                    )
                    ==
                    1;
            }


            // ========================================================
            // KNIGHT
            // ========================================================

            if (
                type == 'n'
                ) {
                return
                    (
                        std::abs(
                            df
                        )
                        ==
                        1
                        &&
                        std::abs(
                            dr
                        )
                        ==
                        2
                        )
                    ||
                    (
                        std::abs(
                            df
                        )
                        ==
                        2
                        &&
                        std::abs(
                            dr
                        )
                        ==
                        1
                        );
            }


            // ========================================================
            // KING
            // ========================================================

            if (
                type == 'k'
                ) {
                return
                    std::abs(
                        df
                    )
                    <=
                    1
                    &&
                    std::abs(
                        dr
                    )
                    <=
                    1
                    &&
                    (
                        df != 0 ||
                        dr != 0
                        );
            }


            // ========================================================
            // SLIDING PIECES
            // ========================================================

            int stepFile = 0;
            int stepRank = 0;


            // Bishop / Queen diagonal.
            if (
                type == 'b' ||
                type == 'q'
                ) {
                if (
                    std::abs(
                        df
                    )
                    ==
                    std::abs(
                        dr
                    )
                    &&
                    df != 0
                    ) {
                    stepFile =
                        df > 0
                        ? 1
                        : -1;

                    stepRank =
                        dr > 0
                        ? 1
                        : -1;
                }
            }


            // Rook / Queen straight.
            if (
                stepFile == 0 &&
                stepRank == 0 &&
                (
                    type == 'r' ||
                    type == 'q'
                    )
                ) {
                if (
                    df == 0 &&
                    dr != 0
                    ) {
                    stepFile =
                        0;

                    stepRank =
                        dr > 0
                        ? 1
                        : -1;
                }

                else if (
                    dr == 0 &&
                    df != 0
                    ) {
                    stepFile =
                        df > 0
                        ? 1
                        : -1;

                    stepRank =
                        0;
                }
            }


            if (
                stepFile == 0 &&
                stepRank == 0
                ) {
                return false;
            }


            int file =
                fromFile +
                stepFile;

            int rank =
                fromRank +
                stepRank;


            while (
                onBoard(
                    file,
                    rank
                )
                ) {
                const int square =
                    rank *
                    8
                    +
                    file;


                if (
                    square ==
                    target
                    ) {
                    return true;
                }


                if (
                    board[
                        square
                    ]
                    !=
                    '.'
                    ) {
                    return false;
                }


                file +=
                    stepFile;

                rank +=
                    stepRank;
            }


            return false;
        }


        // ============================================================
        // SEE SQUARE ATTACKED
        // ============================================================

        bool seeSquareAttacked(
            const SeeBoard& board,
            int target,
            bool byWhite
        ) {
            for (
                int square = 0;
                square < 64;
                ++square
                ) {
                const char piece =
                    board[
                        square
                    ];


                if (
                    piece == '.' ||
                    !pieceBelongsToSide(
                        piece,
                        byWhite
                    )
                    ) {
                    continue;
                }


                if (
                    seePieceAttacksSquare(
                        board,
                        square,
                        target
                    )
                    ) {
                    return true;
                }
            }


            return false;
        }


        // ============================================================
        // SEE KING SQUARE
        // ============================================================

        int seeKingSquare(
            const SeeBoard& board,
            bool white
        ) {
            const char king =
                white
                ? 'K'
                : 'k';


            for (
                int square = 0;
                square < 64;
                ++square
                ) {
                if (
                    board[
                        square
                    ]
                    ==
                    king
                    ) {
                    return square;
                }
            }


            return -1;
        }


        // ============================================================
        // SEE KING SAFETY
        // ============================================================

        bool seeKingSafe(
            const SeeBoard& board,
            bool white
        ) {
            const int kingSquare =
                seeKingSquare(
                    board,
                    white
                );


            if (
                kingSquare < 0
                ) {
                return true;
            }


            return
                !seeSquareAttacked(
                    board,
                    kingSquare,
                    !white
                );
        }


        // ============================================================
        // SEE PROMOTION
        // ============================================================

        char seePromotionPiece(
            char pawn,
            int target
        ) {
            const int rank =
                rankOf(
                    target
                );


            if (
                pawn == 'P' &&
                rank == 7
                ) {
                return 'Q';
            }


            if (
                pawn == 'p' &&
                rank == 0
                ) {
                return 'q';
            }


            return pawn;
        }


        // ============================================================
        // SEE RECAPTURE
        // ============================================================

        int seeRecapture(
            SeeBoard& board,
            int target,
            bool whiteToCapture
        ) {
            const char victim =
                board[
                    target
                ];


            if (
                victim == '.'
                ) {
                return 0;
            }


            int bestGain =
                0;


            for (
                int from = 0;
                from < 64;
                ++from
                ) {
                const char attacker =
                    board[
                        from
                    ];


                if (
                    attacker == '.' ||
                    !pieceBelongsToSide(
                        attacker,
                        whiteToCapture
                    )
                    ) {
                    continue;
                }


                if (
                    !seePieceAttacksSquare(
                        board,
                        from,
                        target
                    )
                    ) {
                    continue;
                }


                const char oldTarget =
                    board[
                        target
                    ];


                const char promotedAttacker =
                    seePromotionPiece(
                        attacker,
                        target
                    );


                board[
                    from
                ] =
                    '.';


                    board[
                        target
                    ] =
                        promotedAttacker;


                        // Reject pinned pieces and illegal king recaptures.
                        if (
                            !seeKingSafe(
                                board,
                                whiteToCapture
                            )
                            ) {
                            board[
                                from
                            ] =
                                attacker;


                                board[
                                    target
                                ] =
                                    oldTarget;


                                    continue;
                        }


                        int promotionBonus =
                            0;


                        if (
                            promotedAttacker !=
                            attacker
                            ) {
                            promotionBonus =
                                pieceValue(
                                    promotedAttacker
                                )
                                -
                                pieceValue(
                                    attacker
                                );
                        }


                        const int gain =
                            pieceValue(
                                oldTarget
                            )
                            +
                            promotionBonus
                            -
                            seeRecapture(
                                board,
                                target,
                                !whiteToCapture
                            );


                        board[
                            from
                        ] =
                            attacker;


                            board[
                                target
                            ] =
                                oldTarget;


                                if (
                                    gain >
                                    bestGain
                                    ) {
                                    bestGain =
                                        gain;
                                }
            }


            return
                std::max(
                    0,
                    bestGain
                );
        }


        // ============================================================
        // SEE SCORE
        // ============================================================

        int staticExchangeEvaluation(
            const Position& pos,
            const Move& move
        ) {
            if (
                !isCapture(
                    pos,
                    move
                )
                &&
                !move.promotion
                ) {
                return 0;
            }


            SeeBoard board =
                pos.board;


            const char movingPiece =
                board[
                    move.from
                ];


            if (
                movingPiece == '.'
                ) {
                return 0;
            }


            const bool movingWhite =
                isWhitePiece(
                    movingPiece
                );


            int capturedSquare =
                move.to;


            char capturedPiece =
                board[
                    move.to
                ];


            // ========================================================
            // EN PASSANT
            // ========================================================

            if (
                move.enPassant
                ) {
                capturedSquare =
                    move.to +
                    (
                        movingWhite
                        ? -8
                        : 8
                        );


                capturedPiece =
                    board[
                        capturedSquare
                    ];


                board[
                    capturedSquare
                ] =
                    '.';
            }


            int immediateGain =
                0;


            if (
                capturedPiece != '.'
                ) {
                immediateGain =
                    pieceValue(
                        capturedPiece
                    );
            }


            board[
                move.from
            ] =
                '.';


                char destinationPiece =
                    movingPiece;


                // ========================================================
                // PROMOTION
                // ========================================================

                if (
                    move.promotion
                    ) {
                    destinationPiece =
                        movingWhite
                        ? static_cast<char>(
                            std::toupper(
                                static_cast<unsigned char>(
                                    move.promotion
                                    )
                            )
                            )
                        : static_cast<char>(
                            std::tolower(
                                static_cast<unsigned char>(
                                    move.promotion
                                    )
                            )
                            );


                    immediateGain +=
                        pieceValue(
                            destinationPiece
                        )
                        -
                        pieceValue(
                            movingPiece
                        );
                }


                board[
                    move.to
                ] =
                    destinationPiece;


                    // Initial move must itself be legal.
                    if (
                        !seeKingSafe(
                            board,
                            movingWhite
                        )
                        ) {
                        return
                            -MATE_SCORE;
                    }


                    const int opponentGain =
                        seeRecapture(
                            board,
                            move.to,
                            !movingWhite
                        );


                    return
                        immediateGain -
                        opponentGain;
        }


        // ============================================================
        // MOVE ORDER SCORE
        // ============================================================

        int moveOrderScore(
            const Position& pos,
            const Move& move,
            const Move* ttMove,
            int ply
        ) {
            int score =
                0;


            // ========================================================
            // TT MOVE
            // ========================================================

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


            // ========================================================
            // PROMOTION
            // ========================================================

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


            // ========================================================
            // CAPTURE + SEE
            // ========================================================

            if (
                capturedPiece != '.'
                ) {
                const int see =
                    staticExchangeEvaluation(
                        pos,
                        move
                    );


                // Winning/equal captures first.
                if (
                    see >= 0
                    ) {
                    score +=
                        25000;
                }

                else {
                    score +=
                        5000;
                }


                // MVV-LVA.
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


                // SEE refinement.
                score +=
                    see *
                    4;
            }


            // ========================================================
            // CASTLING
            // ========================================================

            if (
                move.castle
                ) {
                score +=
                    1000;
            }


            // ========================================================
            // QUIET MOVE HEURISTICS
            // ========================================================

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


        // ============================================================
        // ORDER MOVES
        // ============================================================

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
                scores[
                    i
                ] =
                    moveOrderScore(
                        pos,
                        moves[
                            static_cast<
                                std::size_t
                            >(
                                i
                                )
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
                    int j =
                    i + 1;
                    j <
                    moves.count;
                    ++j
                    ) {
                    if (
                        scores[
                            j
                        ]
                    >
                        scores[
                            bestIndex
                        ]
                        ) {
                        bestIndex =
                            j;
                    }
                }


                if (
                    bestIndex !=
                    i
                    ) {
                    std::swap(
                        scores[
                            i
                        ],
                        scores[
                            bestIndex
                        ]
                    );


                    std::swap(
                        moves[
                            static_cast<
                                std::size_t
                            >(
                                i
                                )
                        ],
                        moves[
                            static_cast<
                                std::size_t
                            >(
                                bestIndex
                                )
                        ]
                    );
                }
            }
        }


        // ============================================================
        // RECORD KILLER / HISTORY
        // ============================================================

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
        // QUIESCENCE SEARCH
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


            // ========================================================
            // MATE / STALEMATE
            // ========================================================

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


            // ========================================================
            // QUIESCENCE LIMIT
            // ========================================================

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
            // IF IN CHECK, SEARCH ALL EVASIONS
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
                        score >=
                        beta
                        ) {
                        return
                            score;
                    }


                    if (
                        score >
                        alpha
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
                standPat >=
                beta
                ) {
                return
                    standPat;
            }


            if (
                standPat >
                alpha
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
                const bool capture =
                    isCapture(
                        pos,
                        move
                    );


                if (
                    !capture &&
                    !move.promotion
                    ) {
                    continue;
                }


                // ====================================================
                // DELTA PRUNING
                // ====================================================

                if (
                    capture &&
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


                // ====================================================
                // SEE PRUNING
                // ====================================================

                const int see =
                    staticExchangeEvaluation(
                        pos,
                        move
                    );


                if (
                    see < 0 &&
                    !move.promotion
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


            // ========================================================
            // SEARCH TACTICAL MOVES
            // ========================================================

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
                    score >=
                    beta
                    ) {
                    return
                        score;
                }


                if (
                    score >
                    alpha
                    ) {
                    alpha =
                        score;
                }
            }


            return alpha;
        }


        // ============================================================
        // NEGAMAX + PVS
        // ============================================================

        int negamax(
            Position& pos,
            int depth,
            int alpha,
            int beta,
            int ply,
            SearchContext& context
        ) {
            // ========================================================
            // MAX PLY SAFETY
            // ========================================================

            if (
                ply >=
                MAX_PLY - 1
                ) {
                return
                    evaluateForSideToMove(
                        pos
                    );
            }


            // ========================================================
            // QUIESCENCE
            // ========================================================

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


            // ========================================================
            // TRANSPOSITION TABLE
            // ========================================================

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
                        alpha >=
                        beta
                        ) {
                        return
                            ttEntry->score;
                    }
                }
            }


            // ========================================================
            // GENERATE LEGAL MOVES
            // ========================================================

            MoveList moves;


            generateLegalMoves(
                pos,
                moves
            );


            // ========================================================
            // CHECKMATE / STALEMATE
            // ========================================================

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


            // ========================================================
            // MOVE ORDERING
            // ========================================================

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


            bool firstMove =
                true;


            // ========================================================
            // PRINCIPAL VARIATION SEARCH
            // ========================================================

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


                int score =
                    0;


                // ====================================================
                // FIRST MOVE: FULL WINDOW
                // ====================================================

                if (
                    firstMove
                    ) {
                    score =
                        -negamax(
                            pos,
                            depth - 1,
                            -beta,
                            -alpha,
                            ply + 1,
                            context
                        );


                    firstMove =
                        false;
                }


                // ====================================================
                // LATER MOVES: PVS SCOUT WINDOW
                // ====================================================

                else {
                    score =
                        -negamax(
                            pos,
                            depth - 1,
                            -alpha - 1,
                            -alpha,
                            ply + 1,
                            context
                        );


                    // =================================================
                    // If it appears to improve alpha,
                    // re-search using the full window.
                    // =================================================

                    if (
                        score >
                        alpha
                        &&
                        score <
                        beta
                        ) {
                        score =
                            -negamax(
                                pos,
                                depth - 1,
                                -beta,
                                -alpha,
                                ply + 1,
                                context
                            );
                    }
                }


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


                // ====================================================
                // BEST SCORE
                // ====================================================

                if (
                    score >
                    bestScore
                    ) {
                    bestScore =
                        score;


                    bestMove =
                        move;
                }


                // ====================================================
                // UPDATE ALPHA
                // ====================================================

                if (
                    score >
                    alpha
                    ) {
                    alpha =
                        score;
                }


                // ====================================================
                // BETA CUTOFF
                // ====================================================

                if (
                    alpha >=
                    beta
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


            // ========================================================
            // TT FLAG
            // ========================================================

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
        // EXTRACT PRINCIPAL VARIATION
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
        // ROOT SEARCH + PVS
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


            // ========================================================
            // NO LEGAL MOVES
            // ========================================================

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


            // ========================================================
            // ORDER ROOT MOVES
            // ========================================================

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


            bool firstMove =
                true;


            // ========================================================
            // ROOT PVS
            // ========================================================

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


                int score =
                    0;


                // ====================================================
                // FIRST MOVE: FULL WINDOW
                // ====================================================

                if (
                    firstMove
                    ) {
                    score =
                        -negamax(
                            pos,
                            depth - 1,
                            -beta,
                            -alpha,
                            1,
                            context
                        );


                    firstMove =
                        false;
                }


                // ====================================================
                // LATER MOVES: SCOUT WINDOW
                // ====================================================

                else {
                    score =
                        -negamax(
                            pos,
                            depth - 1,
                            -alpha - 1,
                            -alpha,
                            1,
                            context
                        );


                    // =================================================
                    // Candidate appears better.
                    // Re-search with full window.
                    // =================================================

                    if (
                        score >
                        alpha
                        &&
                        score <
                        beta
                        ) {
                        score =
                            -negamax(
                                pos,
                                depth - 1,
                                -beta,
                                -alpha,
                                1,
                                context
                            );
                    }
                }


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


            // ========================================================
            // STORE ROOT RESULT
            // ========================================================

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


            // ========================================================
            // NO ROOT MOVES
            // ========================================================

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
                        end -
                        start
                    ).count();


                return bestCompleted;
            }


            bestCompleted.hasMove =
                true;


            bestCompleted.bestMove =
                rootMoves.front();


            // ========================================================
            // ITERATIVE DEEPENING LOOP
            // ========================================================

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


            // ========================================================
            // FINAL TIMING
            // ========================================================

            const auto end =
                std::chrono::steady_clock::now();


            bestCompleted.nodes =
                context.nodes;


            bestCompleted.seconds =
                std::chrono::duration<double>(
                    end -
                    start
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


    // ============================================================
    // TT SIZE
    // ============================================================

    std::size_t transpositionTableSize() {

        return
            ttUsed;
    }


    // ============================================================
    // FIXED DEPTH SEARCH
    // ============================================================

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


    // ============================================================
    // TIMED SEARCH
    // ============================================================

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