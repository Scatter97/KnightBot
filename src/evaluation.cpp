#include "evaluation.hpp"

#include <algorithm>
#include <array>
#include <cctype>

namespace chess {

    namespace {

        // ============================================================
        // TAPERED SCORE
        // ============================================================

        struct TaperedScore {
            int mg = 0;
            int eg = 0;
        };


        // ============================================================
        // GAME PHASE
        // ============================================================

        constexpr int MAX_PHASE = 24;


        // ============================================================
        // MATERIAL VALUES
        // ============================================================

        constexpr int MG_PAWN = 100;
        constexpr int MG_KNIGHT = 320;
        constexpr int MG_BISHOP = 330;
        constexpr int MG_ROOK = 500;
        constexpr int MG_QUEEN = 900;

        constexpr int EG_PAWN = 120;
        constexpr int EG_KNIGHT = 305;
        constexpr int EG_BISHOP = 325;
        constexpr int EG_ROOK = 520;
        constexpr int EG_QUEEN = 900;


        // ============================================================
        // PIECE-SQUARE TABLES
        // ============================================================

        constexpr std::array<int, 64> PAWN_MG = {
             0,  0,  0,  0,  0,  0,  0,  0,
             5, 10, 10,-10,-10, 10, 10,  5,
             5, -5,-10,  0,  0,-10, -5,  5,
             0,  0,  0, 20, 20,  0,  0,  0,
             5,  5, 10, 25, 25, 10,  5,  5,
            10, 10, 20, 30, 30, 20, 10, 10,
            30, 30, 35, 40, 40, 35, 30, 30,
             0,  0,  0,  0,  0,  0,  0,  0
        };


        constexpr std::array<int, 64> PAWN_EG = {
             0,  0,  0,  0,  0,  0,  0,  0,
            10, 10, 10, 10, 10, 10, 10, 10,
            15, 15, 20, 20, 20, 20, 15, 15,
            20, 20, 25, 30, 30, 25, 20, 20,
            30, 30, 35, 40, 40, 35, 30, 30,
            45, 45, 50, 55, 55, 50, 45, 45,
            70, 70, 75, 80, 80, 75, 70, 70,
             0,  0,  0,  0,  0,  0,  0,  0
        };


        constexpr std::array<int, 64> KNIGHT_MG = {
           -50,-40,-30,-30,-30,-30,-40,-50,
           -40,-20,  0,  5,  5,  0,-20,-40,
           -30,  5, 15, 20, 20, 15,  5,-30,
           -30, 10, 20, 30, 30, 20, 10,-30,
           -30, 10, 20, 30, 30, 20, 10,-30,
           -30,  5, 15, 20, 20, 15,  5,-30,
           -40,-20,  0,  5,  5,  0,-20,-40,
           -50,-40,-30,-30,-30,-30,-40,-50
        };


        constexpr std::array<int, 64> KNIGHT_EG = {
           -40,-30,-20,-20,-20,-20,-30,-40,
           -30,-15, -5,  0,  0, -5,-15,-30,
           -20, -5, 10, 15, 15, 10, -5,-20,
           -20,  0, 15, 25, 25, 15,  0,-20,
           -20,  0, 15, 25, 25, 15,  0,-20,
           -20, -5, 10, 15, 15, 10, -5,-20,
           -30,-15, -5,  0,  0, -5,-15,-30,
           -40,-30,-20,-20,-20,-20,-30,-40
        };


        constexpr std::array<int, 64> BISHOP_MG = {
           -20,-10,-10,-10,-10,-10,-10,-20,
           -10,  5,  0,  0,  0,  0,  5,-10,
           -10, 10, 10, 10, 10, 10, 10,-10,
           -10,  0, 10, 20, 20, 10,  0,-10,
           -10,  5, 10, 20, 20, 10,  5,-10,
           -10,  0, 10, 10, 10, 10,  0,-10,
           -10,  5,  0,  0,  0,  0,  5,-10,
           -20,-10,-10,-10,-10,-10,-10,-20
        };


        constexpr std::array<int, 64> BISHOP_EG = {
           -15,-10,-10,-10,-10,-10,-10,-15,
           -10,  0,  0,  0,  0,  0,  0,-10,
           -10,  5, 10, 10, 10, 10,  5,-10,
           -10,  5, 10, 15, 15, 10,  5,-10,
           -10,  5, 10, 15, 15, 10,  5,-10,
           -10,  5, 10, 10, 10, 10,  5,-10,
           -10,  0,  0,  0,  0,  0,  0,-10,
           -15,-10,-10,-10,-10,-10,-10,-15
        };


        constexpr std::array<int, 64> ROOK_MG = {
             0,  0,  5, 10, 10,  5,  0,  0,
            -5,  0,  0,  5,  5,  0,  0, -5,
            -5,  0,  0,  5,  5,  0,  0, -5,
            -5,  0,  0,  5,  5,  0,  0, -5,
            -5,  0,  0,  5,  5,  0,  0, -5,
            -5,  0,  0,  5,  5,  0,  0, -5,
            10, 15, 15, 20, 20, 15, 15, 10,
             0,  0,  5, 10, 10,  5,  0,  0
        };


        constexpr std::array<int, 64> ROOK_EG = {
             0,  0,  5,  5,  5,  5,  0,  0,
             5,  5, 10, 10, 10, 10,  5,  5,
             5,  5, 10, 10, 10, 10,  5,  5,
             5,  5, 10, 15, 15, 10,  5,  5,
             5,  5, 10, 15, 15, 10,  5,  5,
             5,  5, 10, 10, 10, 10,  5,  5,
            10, 10, 15, 15, 15, 15, 10, 10,
             0,  0,  5,  5,  5,  5,  0,  0
        };


        constexpr std::array<int, 64> QUEEN_MG = {
           -20,-10,-10, -5, -5,-10,-10,-20,
           -10,  0,  5,  0,  0,  5,  0,-10,
           -10,  5,  5,  5,  5,  5,  5,-10,
            -5,  0,  5, 10, 10,  5,  0, -5,
             0,  0,  5, 10, 10,  5,  0, -5,
           -10,  5,  5,  5,  5,  5,  5,-10,
           -10,  0,  5,  0,  0,  5,  0,-10,
           -20,-10,-10, -5, -5,-10,-10,-20
        };


        constexpr std::array<int, 64> QUEEN_EG = {
           -10, -5, -5, -5, -5, -5, -5,-10,
            -5,  0,  5,  5,  5,  5,  0, -5,
            -5,  5, 10, 10, 10, 10,  5, -5,
            -5,  5, 10, 15, 15, 10,  5, -5,
            -5,  5, 10, 15, 15, 10,  5, -5,
            -5,  5, 10, 10, 10, 10,  5, -5,
            -5,  0,  5,  5,  5,  5,  0, -5,
           -10, -5, -5, -5, -5, -5, -5,-10
        };


        constexpr std::array<int, 64> KING_MG = {
            20, 30, 10,  0,  0, 10, 30, 20,
            10, 10,  0,-10,-10,  0, 10, 10,
           -10,-20,-20,-20,-20,-20,-20,-10,
           -20,-30,-30,-40,-40,-30,-30,-20,
           -30,-40,-40,-50,-50,-40,-40,-30,
           -30,-40,-40,-50,-50,-40,-40,-30,
           -30,-40,-40,-50,-50,-40,-40,-30,
           -30,-40,-40,-50,-50,-40,-40,-30
        };


        constexpr std::array<int, 64> KING_EG = {
           -40,-30,-20,-20,-20,-20,-30,-40,
           -30,-20,-10,-10,-10,-10,-20,-30,
           -20,-10, 10, 15, 15, 10,-10,-20,
           -20,-10, 15, 25, 25, 15,-10,-20,
           -20,-10, 15, 25, 25, 15,-10,-20,
           -20,-10, 10, 15, 15, 10,-10,-20,
           -30,-20,-10,-10,-10,-10,-20,-30,
           -40,-30,-20,-20,-20,-20,-30,-40
        };


        // ============================================================
        // POSITIONAL BONUSES
        // ============================================================

        constexpr TaperedScore BISHOP_PAIR_BONUS = {
            30,
            45
        };

        constexpr TaperedScore TEMPO_BONUS = {
            10,
            6
        };


        // ============================================================
        // PAWN STRUCTURE VALUES
        // ============================================================

        constexpr TaperedScore ISOLATED_PAWN_PENALTY = {
            -12,
            -16
        };

        constexpr TaperedScore DOUBLED_PAWN_PENALTY = {
            -10,
            -14
        };

        constexpr TaperedScore CONNECTED_PAWN_BONUS = {
            6,
            10
        };

        constexpr TaperedScore PROTECTED_PASSER_BONUS = {
            8,
            18
        };


        // Passed-pawn bonus indexed by relative rank.
        //
        // White:
        // rank 2 -> relative rank 1
        // rank 7 -> relative rank 6
        //
        // Black is mirrored.
        //
        // Index 0 and 7 are unused in normal chess.
        constexpr std::array<int, 8> PASSED_PAWN_MG = {
             0,
             5,
            10,
            18,
            30,
            48,
            75,
             0
        };

        constexpr std::array<int, 8> PASSED_PAWN_EG = {
              0,
             10,
             20,
             35,
             60,
            100,
            160,
              0
        };


        // ============================================================
        // BASIC HELPERS
        // ============================================================

        int fileOf(
            int square
        ) {
            return square & 7;
        }


        int rankOf(
            int square
        ) {
            return square >> 3;
        }


        bool isWhitePiece(
            char piece
        ) {
            return
                piece >= 'A' &&
                piece <= 'Z';
        }


        char pieceType(
            char piece
        ) {
            return
                static_cast<char>(
                    std::tolower(
                        static_cast<unsigned char>(
                            piece
                            )
                    )
                    );
        }


        int mirrorSquare(
            int square
        ) {
            return square ^ 56;
        }


        int tableSquare(
            int square,
            bool white
        ) {
            return
                white
                ? square
                : mirrorSquare(
                    square
                );
        }


        // ============================================================
        // MATERIAL
        // ============================================================

        TaperedScore materialScore(
            char piece
        ) {
            switch (
                pieceType(
                    piece
                )
                ) {
            case 'p':
                return {
                    MG_PAWN,
                    EG_PAWN
                };

            case 'n':
                return {
                    MG_KNIGHT,
                    EG_KNIGHT
                };

            case 'b':
                return {
                    MG_BISHOP,
                    EG_BISHOP
                };

            case 'r':
                return {
                    MG_ROOK,
                    EG_ROOK
                };

            case 'q':
                return {
                    MG_QUEEN,
                    EG_QUEEN
                };

            case 'k':
                return {
                    0,
                    0
                };

            default:
                return {
                    0,
                    0
                };
            }
        }


        // ============================================================
        // PIECE-SQUARE SCORE
        // ============================================================

        TaperedScore pieceSquareScore(
            char piece,
            int square
        ) {
            const bool white =
                isWhitePiece(
                    piece
                );


            const int s =
                tableSquare(
                    square,
                    white
                );


            switch (
                pieceType(
                    piece
                )
                ) {
            case 'p':
                return {
                    PAWN_MG[s],
                    PAWN_EG[s]
                };

            case 'n':
                return {
                    KNIGHT_MG[s],
                    KNIGHT_EG[s]
                };

            case 'b':
                return {
                    BISHOP_MG[s],
                    BISHOP_EG[s]
                };

            case 'r':
                return {
                    ROOK_MG[s],
                    ROOK_EG[s]
                };

            case 'q':
                return {
                    QUEEN_MG[s],
                    QUEEN_EG[s]
                };

            case 'k':
                return {
                    KING_MG[s],
                    KING_EG[s]
                };

            default:
                return {
                    0,
                    0
                };
            }
        }


        // ============================================================
        // PHASE VALUE
        // ============================================================

        int phaseValue(
            char piece
        ) {
            switch (
                pieceType(
                    piece
                )
                ) {
            case 'n':
                return 1;

            case 'b':
                return 1;

            case 'r':
                return 2;

            case 'q':
                return 4;

            default:
                return 0;
            }
        }


        // ============================================================
        // INTERPOLATE
        // ============================================================

        int interpolate(
            int mg,
            int eg,
            int phase
        ) {
            phase =
                std::clamp(
                    phase,
                    0,
                    MAX_PHASE
                );


            return
                (
                    mg *
                    phase
                    +
                    eg *
                    (
                        MAX_PHASE -
                        phase
                        )
                    )
                /
                MAX_PHASE;
        }


        // ============================================================
        // PAWN HELPERS
        // ============================================================

        bool pawnOnFile(
            const Position& pos,
            int file,
            bool white
        ) {
            if (
                file < 0 ||
                file > 7
                ) {
                return false;
            }


            const char pawn =
                white
                ? 'P'
                : 'p';


            for (
                int rank = 0;
                rank < 8;
                ++rank
                ) {
                const int square =
                    rank * 8 +
                    file;


                if (
                    pos.board[square] ==
                    pawn
                    ) {
                    return true;
                }
            }


            return false;
        }


        // ============================================================
        // ISOLATED PAWN
        // ============================================================

        bool isIsolatedPawn(
            const Position& pos,
            int square,
            bool white
        ) {
            const int file =
                fileOf(
                    square
                );


            const bool leftPawn =
                pawnOnFile(
                    pos,
                    file - 1,
                    white
                );


            const bool rightPawn =
                pawnOnFile(
                    pos,
                    file + 1,
                    white
                );


            return
                !leftPawn &&
                !rightPawn;
        }


        // ============================================================
        // DOUBLED PAWN COUNT ON FILE
        // ============================================================

        int pawnCountOnFile(
            const Position& pos,
            int file,
            bool white
        ) {
            const char pawn =
                white
                ? 'P'
                : 'p';


            int count =
                0;


            for (
                int rank = 0;
                rank < 8;
                ++rank
                ) {
                if (
                    pos.board[
                        rank * 8 +
                            file
                    ] ==
                    pawn
                            ) {
                    ++count;
                }
            }


            return count;
        }


        // ============================================================
        // PASSED PAWN
        // ============================================================

        bool isPassedPawn(
            const Position& pos,
            int square,
            bool white
        ) {
            const int file =
                fileOf(
                    square
                );

            const int rank =
                rankOf(
                    square
                );


            const char enemyPawn =
                white
                ? 'p'
                : 'P';


            const int startRank =
                white
                ? rank + 1
                : rank - 1;


            const int endRank =
                white
                ? 7
                : 0;


            const int step =
                white
                ? 1
                : -1;


            for (
                int r = startRank;
                white
                ? r <= endRank
                : r >= endRank;
                r += step
                ) {
                for (
                    int df = -1;
                    df <= 1;
                    ++df
                    ) {
                    const int f =
                        file +
                        df;


                    if (
                        f < 0 ||
                        f > 7
                        ) {
                        continue;
                    }


                    const int target =
                        r * 8 +
                        f;


                    if (
                        pos.board[target] ==
                        enemyPawn
                        ) {
                        return false;
                    }
                }
            }


            return true;
        }


        // ============================================================
        // CONNECTED PAWN
        // ============================================================
        //
        // A pawn counts as connected if a friendly pawn is on an
        // adjacent file on the same rank or one rank behind/ahead.
        //
        // This catches both pawn chains and side-by-side pawns.
        //
        // ============================================================

        bool isConnectedPawn(
            const Position& pos,
            int square,
            bool white
        ) {
            const int file =
                fileOf(
                    square
                );

            const int rank =
                rankOf(
                    square
                );


            const char pawn =
                white
                ? 'P'
                : 'p';


            for (
                int df : { -1, 1 }
                ) {
                const int f =
                    file +
                    df;


                if (
                    f < 0 ||
                    f > 7
                    ) {
                    continue;
                }


                for (
                    int dr = -1;
                    dr <= 1;
                    ++dr
                    ) {
                    const int r =
                        rank +
                        dr;


                    if (
                        r < 0 ||
                        r > 7
                        ) {
                        continue;
                    }


                    const int target =
                        r * 8 +
                        f;


                    if (
                        pos.board[target] ==
                        pawn
                        ) {
                        return true;
                    }
                }
            }


            return false;
        }


        // ============================================================
        // PAWN PROTECTED BY PAWN
        // ============================================================

        bool isPawnProtectedByPawn(
            const Position& pos,
            int square,
            bool white
        ) {
            const int file =
                fileOf(
                    square
                );

            const int rank =
                rankOf(
                    square
                );


            const int protectorRank =
                white
                ? rank - 1
                : rank + 1;


            if (
                protectorRank < 0 ||
                protectorRank > 7
                ) {
                return false;
            }


            const char pawn =
                white
                ? 'P'
                : 'p';


            for (
                int df : { -1, 1 }
                ) {
                const int protectorFile =
                    file +
                    df;


                if (
                    protectorFile < 0 ||
                    protectorFile > 7
                    ) {
                    continue;
                }


                const int protectorSquare =
                    protectorRank * 8 +
                    protectorFile;


                if (
                    pos.board[
                        protectorSquare
                    ] ==
                    pawn
                            ) {
                    return true;
                }
            }


            return false;
        }


        // ============================================================
        // RELATIVE PAWN RANK
        // ============================================================

        int relativePawnRank(
            int square,
            bool white
        ) {
            const int rank =
                rankOf(
                    square
                );


            return
                white
                ? rank
                : 7 - rank;
        }


        // ============================================================
        // PAWN STRUCTURE EVALUATION
        // ============================================================

        TaperedScore evaluatePawnStructure(
            const Position& pos
        ) {
            TaperedScore score;


            // ========================================================
            // DOUBLED PAWNS
            // ========================================================
            //
            // Penalize each extra pawn on a file.
            //
            // 1 pawn = no penalty
            // 2 pawns = one penalty
            // 3 pawns = two penalties
            //
            // ========================================================

            for (
                int file = 0;
                file < 8;
                ++file
                ) {
                const int whiteCount =
                    pawnCountOnFile(
                        pos,
                        file,
                        true
                    );


                const int blackCount =
                    pawnCountOnFile(
                        pos,
                        file,
                        false
                    );


                if (
                    whiteCount > 1
                    ) {
                    const int extras =
                        whiteCount -
                        1;


                    score.mg +=
                        DOUBLED_PAWN_PENALTY.mg *
                        extras;

                    score.eg +=
                        DOUBLED_PAWN_PENALTY.eg *
                        extras;
                }


                if (
                    blackCount > 1
                    ) {
                    const int extras =
                        blackCount -
                        1;


                    // Black penalty helps White.
                    score.mg -=
                        DOUBLED_PAWN_PENALTY.mg *
                        extras;

                    score.eg -=
                        DOUBLED_PAWN_PENALTY.eg *
                        extras;
                }
            }


            // ========================================================
            // PER-PAWN FEATURES
            // ========================================================

            for (
                int square = 0;
                square < 64;
                ++square
                ) {
                const char piece =
                    pos.board[
                        square
                    ];


                if (
                    piece != 'P' &&
                    piece != 'p'
                    ) {
                    continue;
                }


                const bool white =
                    piece == 'P';


                const int sign =
                    white
                    ? 1
                    : -1;


                // ====================================================
                // ISOLATED
                // ====================================================

                if (
                    isIsolatedPawn(
                        pos,
                        square,
                        white
                    )
                    ) {
                    score.mg +=
                        sign *
                        ISOLATED_PAWN_PENALTY.mg;

                    score.eg +=
                        sign *
                        ISOLATED_PAWN_PENALTY.eg;
                }


                // ====================================================
                // CONNECTED
                // ====================================================

                if (
                    isConnectedPawn(
                        pos,
                        square,
                        white
                    )
                    ) {
                    score.mg +=
                        sign *
                        CONNECTED_PAWN_BONUS.mg;

                    score.eg +=
                        sign *
                        CONNECTED_PAWN_BONUS.eg;
                }


                // ====================================================
                // PASSED PAWN
                // ====================================================

                if (
                    isPassedPawn(
                        pos,
                        square,
                        white
                    )
                    ) {
                    const int relativeRank =
                        std::clamp(
                            relativePawnRank(
                                square,
                                white
                            ),
                            0,
                            7
                        );


                    score.mg +=
                        sign *
                        PASSED_PAWN_MG[
                            relativeRank
                        ];


                    score.eg +=
                        sign *
                        PASSED_PAWN_EG[
                            relativeRank
                        ];


                    // ================================================
                    // PROTECTED PASSED PAWN
                    // ================================================

                    if (
                        isPawnProtectedByPawn(
                            pos,
                            square,
                            white
                        )
                        ) {
                        score.mg +=
                            sign *
                            PROTECTED_PASSER_BONUS.mg;

                        score.eg +=
                            sign *
                            PROTECTED_PASSER_BONUS.eg;
                    }
                }
            }


            return score;
        }

    } // anonymous namespace


    // ============================================================
    // PUBLIC PIECE VALUE
    // ============================================================

    int pieceValue(
        char piece
    ) {
        switch (
            pieceType(
                piece
            )
            ) {
        case 'p':
            return 100;

        case 'n':
            return 320;

        case 'b':
            return 330;

        case 'r':
            return 500;

        case 'q':
            return 900;

        case 'k':
            return 20000;

        default:
            return 0;
        }
    }


    // ============================================================
    // FULL STATIC EVALUATION
    // ============================================================

    int evaluate(
        const Position& pos
    ) {
        // ========================================================
 // INSUFFICIENT MATERIAL
 // ========================================================

        if (
            isInsufficientMaterial(
                pos
            )
            ) {
            return 0;
        }
        int mgScore =
            0;

        int egScore =
            0;

        int phase =
            0;


        int whiteBishops =
            0;

        int blackBishops =
            0;


        // ========================================================
        // MATERIAL + PIECE-SQUARE TABLES
        // ========================================================

        for (
            int square = 0;
            square < 64;
            ++square
            ) {
            const char piece =
                pos.board[
                    square
                ];


            if (
                piece == '.'
                ) {
                continue;
            }


            const bool white =
                isWhitePiece(
                    piece
                );


            const int sign =
                white
                ? 1
                : -1;


            const TaperedScore material =
                materialScore(
                    piece
                );


            const TaperedScore positional =
                pieceSquareScore(
                    piece,
                    square
                );


            mgScore +=
                sign *
                (
                    material.mg +
                    positional.mg
                    );


            egScore +=
                sign *
                (
                    material.eg +
                    positional.eg
                    );


            phase +=
                phaseValue(
                    piece
                );


            if (
                piece == 'B'
                ) {
                ++whiteBishops;
            }

            else if (
                piece == 'b'
                ) {
                ++blackBishops;
            }
        }


        // ========================================================
        // BISHOP PAIR
        // ========================================================

        if (
            whiteBishops >= 2
            ) {
            mgScore +=
                BISHOP_PAIR_BONUS.mg;

            egScore +=
                BISHOP_PAIR_BONUS.eg;
        }


        if (
            blackBishops >= 2
            ) {
            mgScore -=
                BISHOP_PAIR_BONUS.mg;

            egScore -=
                BISHOP_PAIR_BONUS.eg;
        }


        // ========================================================
        // PAWN STRUCTURE
        // ========================================================

        const TaperedScore pawnStructure =
            evaluatePawnStructure(
                pos
            );


        mgScore +=
            pawnStructure.mg;

        egScore +=
            pawnStructure.eg;


        // ========================================================
        // TEMPO
        // ========================================================

        if (
            pos.whiteToMove
            ) {
            mgScore +=
                TEMPO_BONUS.mg;

            egScore +=
                TEMPO_BONUS.eg;
        }

        else {
            mgScore -=
                TEMPO_BONUS.mg;

            egScore -=
                TEMPO_BONUS.eg;
        }


        // ========================================================
        // FINAL TAPER
        // ========================================================

        return
            interpolate(
                mgScore,
                egScore,
                phase
            );
    }


    // ============================================================
    // SIDE-TO-MOVE EVALUATION
    // ============================================================

    int evaluateForSideToMove(
        const Position& pos
    ) {
        const int whiteScore =
            evaluate(
                pos
            );


        return
            pos.whiteToMove
            ? whiteScore
            : -whiteScore;
    }

} // namespace chess