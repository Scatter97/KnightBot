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

        constexpr int ASPIRATION_INITIAL_WINDOW = 25;
        constexpr int ASPIRATION_MIN_DEPTH = 4;

        constexpr int NULL_MOVE_MIN_DEPTH = 3;
        constexpr int NULL_MOVE_BASE_REDUCTION = 2;

        // LMR
        constexpr int LMR_MIN_DEPTH = 3;
        constexpr int LMR_MIN_MOVE_INDEX = 4;


        // ============================================================
        // FIXED-SIZE TRANSPOSITION TABLE
        // ============================================================

        constexpr std::size_t TT_SIZE = 1ULL << 20;
        constexpr std::size_t TT_MASK = TT_SIZE - 1;

        enum class TTFlag : std::uint8_t {
            Exact,
            LowerBound,
            UpperBound
        };

        struct TTEntry {
            std::uint64_t key = 0;

            int score = 0;
            int depth = -1;

            TTFlag flag = TTFlag::Exact;

            Move bestMove{};

            bool valid = false;
        };

        std::array<TTEntry, TT_SIZE> transpositionTable{};

        std::size_t ttUsed = 0;


        // ============================================================
        // TRANSPOSITION TABLE
        // ============================================================

        TTEntry* probeTT(
            std::uint64_t key
        ) {
            TTEntry& entry =
                transpositionTable[
                    key & TT_MASK
                ];

            if (
                entry.valid &&
                entry.key == key
                ) {
                return &entry;
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
                    key & TT_MASK
                ];

            if (!entry.valid) {
                ++ttUsed;
            }

            if (
                !entry.valid ||
                entry.key != key ||
                depth >= entry.depth
                ) {
                entry.key = key;
                entry.depth = depth;
                entry.score = score;
                entry.flag = flag;
                entry.bestMove = move;
                entry.valid = true;
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
            std::uint64_t nodes = 0;

            bool useDeadline = false;

            std::chrono::steady_clock::time_point deadline{};

            bool stopped = false;
        };


        // ============================================================
        // BASIC HELPERS
        // ============================================================

        int fileOf(int square) {
            return square & 7;
        }


        int rankOf(int square) {
            return square >> 3;
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


        bool isWhitePiece(char piece) {
            return
                piece >= 'A' &&
                piece <= 'Z';
        }


        bool isBlackPiece(char piece) {
            return
                piece >= 'a' &&
                piece <= 'z';
        }


        bool pieceBelongsToSide(
            char piece,
            bool white
        ) {
            return
                white
                ? isWhitePiece(piece)
                : isBlackPiece(piece);
        }


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
                pos.board[move.to] != '.';
        }


        char capturedPieceForMove(
            const Position& pos,
            const Move& move
        ) {
            if (move.enPassant) {
                return
                    pos.whiteToMove
                    ? 'p'
                    : 'P';
            }

            return pos.board[move.to];
        }


        // ============================================================
        // TIME
        // ============================================================

        bool timeExpired(
            SearchContext& context
        ) {
            if (!context.useDeadline) {
                return false;
            }

            if (
                (context.nodes & 4095ULL) != 0
                ) {
                return false;
            }

            if (
                std::chrono::steady_clock::now() >=
                context.deadline
                ) {
                context.stopped = true;
                return true;
            }

            return false;
        }


        // ============================================================
        // STATIC EXCHANGE EVALUATION
        // ============================================================

        using SeeBoard =
            std::array<char, 64>;


        bool seePieceAttacksSquare(
            const SeeBoard& board,
            int from,
            int target
        ) {
            const char piece =
                board[from];

            if (piece == '.') {
                return false;
            }

            const int fromFile =
                fileOf(from);

            const int fromRank =
                rankOf(from);

            const int targetFile =
                fileOf(target);

            const int targetRank =
                rankOf(target);

            const int df =
                targetFile - fromFile;

            const int dr =
                targetRank - fromRank;

            const char type =
                static_cast<char>(
                    std::tolower(
                        static_cast<unsigned char>(
                            piece
                            )
                    )
                    );


            if (type == 'p') {
                if (isWhitePiece(piece)) {
                    return
                        dr == 1 &&
                        std::abs(df) == 1;
                }

                return
                    dr == -1 &&
                    std::abs(df) == 1;
            }


            if (type == 'n') {
                return
                    (
                        std::abs(df) == 1 &&
                        std::abs(dr) == 2
                        )
                    ||
                    (
                        std::abs(df) == 2 &&
                        std::abs(dr) == 1
                        );
            }


            if (type == 'k') {
                return
                    std::abs(df) <= 1 &&
                    std::abs(dr) <= 1 &&
                    (
                        df != 0 ||
                        dr != 0
                        );
            }


            int stepFile = 0;
            int stepRank = 0;


            if (
                type == 'b' ||
                type == 'q'
                ) {
                if (
                    std::abs(df) ==
                    std::abs(dr)
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
                    stepFile = 0;

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

                    stepRank = 0;
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
                    rank * 8 +
                    file;

                if (
                    square == target
                    ) {
                    return true;
                }

                if (
                    board[square] != '.'
                    ) {
                    return false;
                }

                file += stepFile;
                rank += stepRank;
            }

            return false;
        }


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
                    board[square];

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
                    board[square] == king
                    ) {
                    return square;
                }
            }

            return -1;
        }


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


        char seePromotionPiece(
            char pawn,
            int target
        ) {
            const int rank =
                rankOf(target);

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
                board[target];

            if (
                victim == '.'
                ) {
                return 0;
            }

            int bestGain = 0;


            for (
                int from = 0;
                from < 64;
                ++from
                ) {
                const char attacker =
                    board[from];

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
                    board[target];

                const char promotedAttacker =
                    seePromotionPiece(
                        attacker,
                        target
                    );


                board[from] = '.';

                board[target] =
                    promotedAttacker;


                if (
                    !seeKingSafe(
                        board,
                        whiteToCapture
                    )
                    ) {
                    board[from] =
                        attacker;

                    board[target] =
                        oldTarget;

                    continue;
                }


                int promotionBonus = 0;

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


                board[from] =
                    attacker;

                board[target] =
                    oldTarget;


                if (
                    gain > bestGain
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
                board[move.from];

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
                board[move.to];


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
                    board[capturedSquare];

                board[capturedSquare] =
                    '.';
            }


            int immediateGain = 0;

            if (
                capturedPiece != '.'
                ) {
                immediateGain =
                    pieceValue(
                        capturedPiece
                    );
            }


            board[move.from] =
                '.';


            char destinationPiece =
                movingPiece;


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


            board[move.to] =
                destinationPiece;


            if (
                !seeKingSafe(
                    board,
                    movingWhite
                )
                ) {
                return -MATE_SCORE;
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
                score += 1000000;
            }


            const char movingPiece =
                pos.board[move.from];

            const char capturedPiece =
                capturedPieceForMove(
                    pos,
                    move
                );


            if (
                move.promotion
                ) {
                score += 30000;

                score +=
                    pieceValue(
                        move.promotion
                    );
            }


            if (
                capturedPiece != '.'
                ) {
                const int see =
                    staticExchangeEvaluation(
                        pos,
                        move
                    );

                if (
                    see >= 0
                    ) {
                    score += 25000;
                }

                else {
                    score += 5000;
                }

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

                score +=
                    see *
                    4;
            }


            if (
                move.castle
                ) {
                score += 1000;
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
                        killerMoves[ply][0]
                    )
                    &&
                    sameMove(
                        move,
                        killerMoves[ply][0]
                    )
                    ) {
                    score += 4000;
                }

                else if (
                    validStoredMove(
                        killerMoves[ply][1]
                    )
                    &&
                    sameMove(
                        move,
                        killerMoves[ply][1]
                    )
                    ) {
                    score += 3000;
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
                i < moves.count - 1;
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


        // ============================================================
        // KILLER / HISTORY UPDATE
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
                    killerMoves[ply][0]
                )
                ) {
                killerMoves[ply][1] =
                    killerMoves[ply][0];

                killerMoves[ply][0] =
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
                        history > 100000
                        ) {
                        history /= 2;
                    }
        }


        // ============================================================
        // NULL MOVE
        // ============================================================

        struct NullMoveUndo {
            bool whiteToMove = true;

            int enPassantSquare = -1;

            int halfmoveClock = 0;
            int fullmoveNumber = 1;

            std::uint64_t zobristKey = 0;
        };


        bool hasNonPawnMaterial(
            const Position& pos,
            bool white
        ) {
            if (white) {
                return
                    pos.pieces[WN] != 0 ||
                    pos.pieces[WB] != 0 ||
                    pos.pieces[WR] != 0 ||
                    pos.pieces[WQ] != 0;
            }

            return
                pos.pieces[BN] != 0 ||
                pos.pieces[BB] != 0 ||
                pos.pieces[BR] != 0 ||
                pos.pieces[BQ] != 0;
        }


        void makeNullMove(
            Position& pos,
            NullMoveUndo& undo
        ) {
            undo.whiteToMove =
                pos.whiteToMove;

            undo.enPassantSquare =
                pos.enPassantSquare;

            undo.halfmoveClock =
                pos.halfmoveClock;

            undo.fullmoveNumber =
                pos.fullmoveNumber;

            undo.zobristKey =
                pos.zobristKey;


            const bool wasWhite =
                pos.whiteToMove;


            pos.whiteToMove =
                !pos.whiteToMove;

            pos.enPassantSquare =
                -1;

            ++pos.halfmoveClock;


            if (!wasWhite) {
                ++pos.fullmoveNumber;
            }


            pos.zobristKey =
                calculateZobrist(
                    pos
                );
        }


        void undoNullMove(
            Position& pos,
            const NullMoveUndo& undo
        ) {
            pos.whiteToMove =
                undo.whiteToMove;

            pos.enPassantSquare =
                undo.enPassantSquare;

            pos.halfmoveClock =
                undo.halfmoveClock;

            pos.fullmoveNumber =
                undo.fullmoveNumber;

            pos.zobristKey =
                undo.zobristKey;
        }


        // ============================================================
        // LMR REDUCTION
        // ============================================================

        int lateMoveReduction(
            int depth,
            int moveIndex
        ) {
            int reduction = 1;

            if (
                depth >= 6 &&
                moveIndex >= 8
                ) {
                reduction = 2;
            }

            if (
                depth >= 10 &&
                moveIndex >= 12
                ) {
                reduction = 3;
            }


            // Always leave at least one ply to search.
            reduction =
                std::min(
                    reduction,
                    depth - 2
                );


            return
                std::max(
                    1,
                    reduction
                );
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
                if (checked) {
                    return
                        -MATE_SCORE +
                        ply;
                }

                return 0;
            }


            if (
                qply >= MAX_QPLY
                ) {
                return
                    evaluateForSideToMove(
                        pos
                    );
            }


            // ========================================================
            // CHECK EVASIONS
            // ========================================================

            if (checked) {
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
                        alpha = score;
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
                return standPat;
            }


            if (
                standPat > alpha
                ) {
                alpha = standPat;
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
                    alpha = score;
                }
            }


            return alpha;
        }


        // ============================================================
        // NEGAMAX + PVS + NULL MOVE + LMR
        // ============================================================

        int negamax(
            Position& pos,
            int depth,
            int alpha,
            int beta,
            int ply,
            SearchContext& context,
            bool allowNull
        ) {
            if (
                ply >= MAX_PLY - 1
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
            bool hasTTMove = false;


            TTEntry* ttEntry =
                probeTT(
                    key
                );


            // ========================================================
            // TT
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
                    ttEntry->depth >= depth
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


            const bool checked =
                inCheck(
                    pos,
                    pos.whiteToMove
                );


            // ========================================================
            // NULL-MOVE PRUNING
            // ========================================================

            if (
                allowNull &&
                !checked &&
                depth >= NULL_MOVE_MIN_DEPTH &&
                hasNonPawnMaterial(
                    pos,
                    pos.whiteToMove
                ) &&
                beta < MATE_THRESHOLD
                ) {
                NullMoveUndo nullUndo;


                makeNullMove(
                    pos,
                    nullUndo
                );


                const int reduction =
                    NULL_MOVE_BASE_REDUCTION
                    +
                    depth / 6;


                const int nullDepth =
                    std::max(
                        0,
                        depth -
                        1 -
                        reduction
                    );


                const int nullScore =
                    -negamax(
                        pos,
                        nullDepth,
                        -beta,
                        -beta + 1,
                        ply + 1,
                        context,
                        false
                    );


                undoNullMove(
                    pos,
                    nullUndo
                );


                if (
                    context.stopped
                    ) {
                    return 0;
                }


                if (
                    nullScore >= beta
                    ) {
                    if (
                        nullScore >=
                        MATE_THRESHOLD
                        ) {
                        return beta;
                    }


                    return nullScore;
                }
            }


            // ========================================================
            // LEGAL MOVES
            // ========================================================

            MoveList moves;


            generateLegalMoves(
                pos,
                moves
            );


            if (
                moves.empty()
                ) {
                if (checked) {
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


            bool firstMove =
                true;

            int moveIndex =
                0;


            // ========================================================
            // SEARCH MOVES
            // ========================================================

            for (
                const Move& move :
                moves
                ) {
                const bool capture =
                    isCapture(
                        pos,
                        move
                    );


                const bool promotion =
                    move.promotion != 0;


                const bool isTTMove =
                    hasTTMove &&
                    sameMove(
                        move,
                        ttMove
                    );


                UndoState undo;


                makeMove(
                    pos,
                    move,
                    undo
                );


                // After makeMove(), pos.whiteToMove is the opponent.
                // Therefore if the opponent is in check, this move gave check.
                const bool givesCheck =
                    inCheck(
                        pos,
                        pos.whiteToMove
                    );


                int score = 0;


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
                            context,
                            true
                        );


                    firstMove =
                        false;
                }


                // ====================================================
                // LATER MOVES: PVS + LMR
                // ====================================================

                else {
                    const bool quiet =
                        !capture &&
                        !promotion;


                    const bool canReduce =
                        depth >=
                        LMR_MIN_DEPTH
                        &&
                        moveIndex >=
                        LMR_MIN_MOVE_INDEX
                        &&
                        quiet
                        &&
                        !checked
                        &&
                        !givesCheck
                        &&
                        !isTTMove;


                    // =================================================
                    // LMR
                    // =================================================

                    if (
                        canReduce
                        ) {
                        const int reduction =
                            lateMoveReduction(
                                depth,
                                moveIndex
                            );


                        const int reducedDepth =
                            std::max(
                                0,
                                depth -
                                1 -
                                reduction
                            );


                        // Reduced scout search.
                        score =
                            -negamax(
                                pos,
                                reducedDepth,
                                -alpha - 1,
                                -alpha,
                                ply + 1,
                                context,
                                true
                            );


                        // If this supposedly unimportant move actually
                        // beats alpha, restore full depth and verify it.
                        if (
                            score > alpha
                            ) {
                            score =
                                -negamax(
                                    pos,
                                    depth - 1,
                                    -alpha - 1,
                                    -alpha,
                                    ply + 1,
                                    context,
                                    true
                                );


                            // If the normal scout still beats alpha,
                            // obtain its exact score.
                            if (
                                score > alpha &&
                                score < beta
                                ) {
                                score =
                                    -negamax(
                                        pos,
                                        depth - 1,
                                        -beta,
                                        -alpha,
                                        ply + 1,
                                        context,
                                        true
                                    );
                            }
                        }
                    }


                    // =================================================
                    // NORMAL PVS
                    // =================================================

                    else {
                        score =
                            -negamax(
                                pos,
                                depth - 1,
                                -alpha - 1,
                                -alpha,
                                ply + 1,
                                context,
                                true
                            );


                        if (
                            score > alpha &&
                            score < beta
                            ) {
                            score =
                                -negamax(
                                    pos,
                                    depth - 1,
                                    -beta,
                                    -alpha,
                                    ply + 1,
                                    context,
                                    true
                                );
                        }
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


                if (
                    score > bestScore
                    ) {
                    bestScore =
                        score;

                    bestMove =
                        move;
                }


                if (
                    score > alpha
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


                ++moveIndex;
            }


            // ========================================================
            // TT STORE
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
        // PRINCIPAL VARIATION
        // ============================================================

        std::vector<Move> extractPV(
            Position pos,
            int maxDepth
        ) {
            std::vector<Move> pv;


            pv.reserve(
                static_cast<std::size_t>(
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


                if (!found) {
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
            int alpha,
            int beta,
            SearchContext& context
        ) {
            SearchResult result;


            result.depth =
                depth;


            const int originalAlpha =
                alpha;

            const int originalBeta =
                beta;


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
            bool hasTTMove = false;


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


            int bestScore =
                -INF;


            Move bestMove =
                moves.front();


            bool firstMove =
                true;


            // Root stays full depth.
            // We only apply LMR below the root.
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


                int score = 0;


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
                            context,
                            true
                        );


                    firstMove =
                        false;
                }

                else {
                    score =
                        -negamax(
                            pos,
                            depth - 1,
                            -alpha - 1,
                            -alpha,
                            1,
                            context,
                            true
                        );


                    if (
                        score > alpha &&
                        score < beta
                        ) {
                        score =
                            -negamax(
                                pos,
                                depth - 1,
                                -beta,
                                -alpha,
                                1,
                                context,
                                true
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
                    score > bestScore
                    ) {
                    bestScore =
                        score;

                    bestMove =
                        move;
                }


                if (
                    score > alpha
                    ) {
                    alpha =
                        score;
                }


                if (
                    alpha >= beta
                    ) {
                    break;
                }
            }


            if (
                !context.stopped
                ) {
                result.bestMove =
                    bestMove;

                result.score =
                    bestScore;


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
                    rootKey,
                    depth,
                    bestScore,
                    flag,
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
        // ITERATIVE DEEPENING + ASPIRATION WINDOWS
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


            int previousScore =
                0;


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


                SearchResult current;


                // ====================================================
                // FULL WINDOW AT LOW DEPTH
                // ====================================================

                if (
                    depth <
                    ASPIRATION_MIN_DEPTH
                    ) {
                    current =
                        searchRoot(
                            pos,
                            depth,
                            -INF,
                            INF,
                            context
                        );
                }


                // ====================================================
                // ASPIRATION WINDOW
                // ====================================================

                else {
                    int window =
                        ASPIRATION_INITIAL_WINDOW;


                    int alpha =
                        std::max(
                            -INF,
                            previousScore -
                            window
                        );


                    int beta =
                        std::min(
                            INF,
                            previousScore +
                            window
                        );


                    while (true) {
                        current =
                            searchRoot(
                                pos,
                                depth,
                                alpha,
                                beta,
                                context
                            );


                        if (
                            context.stopped
                            ) {
                            break;
                        }


                        // Fail low.
                        if (
                            current.score <= alpha
                            ) {
                            window *= 2;


                            if (
                                window >= INF
                                ) {
                                alpha = -INF;
                                beta = INF;
                            }

                            else {
                                alpha =
                                    std::max(
                                        -INF,
                                        previousScore -
                                        window
                                    );

                                beta =
                                    std::min(
                                        INF,
                                        previousScore +
                                        window
                                    );
                            }


                            continue;
                        }


                        // Fail high.
                        if (
                            current.score >= beta
                            ) {
                            window *= 2;


                            if (
                                window >= INF
                                ) {
                                alpha = -INF;
                                beta = INF;
                            }

                            else {
                                alpha =
                                    std::max(
                                        -INF,
                                        previousScore -
                                        window
                                    );

                                beta =
                                    std::min(
                                        INF,
                                        previousScore +
                                        window
                                    );
                            }


                            continue;
                        }


                        break;
                    }
                }


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


                previousScore =
                    current.score;


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


        ttUsed = 0;

        killerMoves = {};

        historyTable = {};
    }


    std::size_t transpositionTableSize() {
        return ttUsed;
    }


    SearchResult searchBestMove(
        const Position& pos,
        int maxDepth
    ) {
        if (
            maxDepth < 1
            ) {
            maxDepth = 1;
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
            milliseconds = 1;
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