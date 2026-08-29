#include "evaluation.hpp"
#include "nnue.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>

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
// MOBILITY BONUSES
// ============================================================
//
// Bonus for each legal-looking destination square available to
// the piece. This is pseudo-legal mobility; king safety is not
// checked here because this is static evaluation.
//
// ============================================================

        constexpr TaperedScore KNIGHT_MOBILITY_BONUS = {
            4,
            4
        };

        constexpr TaperedScore BISHOP_MOBILITY_BONUS = {
            4,
            5
        };

        constexpr TaperedScore ROOK_MOBILITY_BONUS = {
            2,
            4
        };

        constexpr TaperedScore QUEEN_MOBILITY_BONUS = {
            1,
            2
        };


        // ============================================================
        // ROOK ACTIVITY
        // ============================================================

        constexpr TaperedScore ROOK_SEMI_OPEN_FILE_BONUS = {
            12,
            8
        };

        constexpr TaperedScore ROOK_OPEN_FILE_BONUS = {
            22,
            14
        };

        constexpr TaperedScore ROOK_SEVENTH_RANK_BONUS = {
            20,
            30
        };
        // ============================================================
// KING SAFETY
// ============================================================

        constexpr TaperedScore KING_SHIELD_PAWN_BONUS = {
            12,
            0
        };

        constexpr TaperedScore KING_MISSING_SHIELD_PENALTY = {
            -10,
            0
        };

        constexpr TaperedScore KING_SEMI_OPEN_FILE_PENALTY = {
            -10,
            0
        };

        constexpr TaperedScore KING_OPEN_FILE_PENALTY = {
            -18,
            0
        };

        constexpr TaperedScore KING_CENTER_FILE_PENALTY = {
            -8,
            0
        };
        // ============================================================
// KING ATTACK PRESSURE
// ============================================================
//
// These are middlegame-only because direct king attacks matter
// much less once most attacking material has disappeared.
//
// Value is applied for each square in the king zone attacked
// by that enemy piece.
//

        constexpr int KING_ATTACK_KNIGHT_WEIGHT = 6;
        constexpr int KING_ATTACK_BISHOP_WEIGHT = 5;
        constexpr int KING_ATTACK_ROOK_WEIGHT = 7;
        constexpr int KING_ATTACK_QUEEN_WEIGHT = 10;


        // Extra penalty for having several different pieces attacking
        // the king zone at once.
        constexpr int KING_MULTI_ATTACKER_BONUS = 8;
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
// FRIENDLY PIECE
// ============================================================

        bool friendlyPiece(
            char piece,
            bool white
        ) {
            if (
                piece == '.'
                ) {
                return false;
            }


            return
                isWhitePiece(
                    piece
                ) ==
                white;
        }


        // ============================================================
        // KNIGHT MOBILITY
        // ============================================================

        int knightMobility(
            const Position& pos,
            int square,
            bool white
        ) {
            constexpr std::array<
                std::array<int, 2>,
                8
            > offsets = { {
                { 1,  2 },
                { 2,  1 },
                { 2, -1 },
                { 1, -2 },
                {-1, -2 },
                {-2, -1 },
                {-2,  1 },
                {-1,  2 }
            } };


            const int file =
                fileOf(
                    square
                );

            const int rank =
                rankOf(
                    square
                );


            int mobility =
                0;


            for (
                const auto& offset :
                offsets
                ) {
                const int targetFile =
                    file +
                    offset[0];

                const int targetRank =
                    rank +
                    offset[1];


                if (
                    targetFile < 0 ||
                    targetFile > 7 ||
                    targetRank < 0 ||
                    targetRank > 7
                    ) {
                    continue;
                }


                const int targetSquare =
                    targetRank *
                    8 +
                    targetFile;


                if (
                    !friendlyPiece(
                        pos.board[
                            targetSquare
                        ],
                        white
                    )
                    ) {
                    ++mobility;
                }
            }


            return mobility;
        }


        // ============================================================
        // SLIDING MOBILITY
        // ============================================================

        int slidingMobility(
            const Position& pos,
            int square,
            bool white,
            const std::array<
            std::array<int, 2>,
            8
            >& directions,
            int directionCount
        ) {
            const int startFile =
                fileOf(
                    square
                );

            const int startRank =
                rankOf(
                    square
                );


            int mobility =
                0;


            for (
                int direction = 0;
                direction <
                directionCount;
                ++direction
                ) {
                const int df =
                    directions[
                        direction
                    ][0];

                const int dr =
                    directions[
                        direction
                    ][1];


                int file =
                    startFile +
                    df;

                int rank =
                    startRank +
                    dr;


                while (
                    file >= 0 &&
                    file < 8 &&
                    rank >= 0 &&
                    rank < 8
                    ) {
                    const int targetSquare =
                        rank *
                        8 +
                        file;


                    const char targetPiece =
                        pos.board[
                            targetSquare
                        ];


                    if (
                        targetPiece == '.'
                        ) {
                        ++mobility;
                    }

                    else {
                        if (
                            !friendlyPiece(
                                targetPiece,
                                white
                            )
                            ) {
                            ++mobility;
                        }


                        break;
                    }


                    file +=
                        df;

                    rank +=
                        dr;
                }
            }


            return mobility;
        }


        // ============================================================
        // BISHOP MOBILITY
        // ============================================================

        int bishopMobility(
            const Position& pos,
            int square,
            bool white
        ) {
            constexpr std::array<
                std::array<int, 2>,
                8
            > directions = { {
                { 1,  1 },
                {-1,  1 },
                { 1, -1 },
                {-1, -1 },
                { 0,  0 },
                { 0,  0 },
                { 0,  0 },
                { 0,  0 }
            } };


            return
                slidingMobility(
                    pos,
                    square,
                    white,
                    directions,
                    4
                );
        }


        // ============================================================
        // ROOK MOBILITY
        // ============================================================

        int rookMobility(
            const Position& pos,
            int square,
            bool white
        ) {
            constexpr std::array<
                std::array<int, 2>,
                8
            > directions = { {
                { 1,  0 },
                {-1,  0 },
                { 0,  1 },
                { 0, -1 },
                { 0,  0 },
                { 0,  0 },
                { 0,  0 },
                { 0,  0 }
            } };


            return
                slidingMobility(
                    pos,
                    square,
                    white,
                    directions,
                    4
                );
        }


        // ============================================================
        // QUEEN MOBILITY
        // ============================================================

        int queenMobility(
            const Position& pos,
            int square,
            bool white
        ) {
            constexpr std::array<
                std::array<int, 2>,
                8
            > directions = { {
                { 1,  0 },
                {-1,  0 },
                { 0,  1 },
                { 0, -1 },
                { 1,  1 },
                {-1,  1 },
                { 1, -1 },
                {-1, -1 }
            } };


            return
                slidingMobility(
                    pos,
                    square,
                    white,
                    directions,
                    8
                );
        }


        // ============================================================
        // FILE PAWN INFORMATION
        // ============================================================

        bool fileHasWhitePawn(
            const Position& pos,
            int file
        ) {
            return
                pawnOnFile(
                    pos,
                    file,
                    true
                );
        }


        bool fileHasBlackPawn(
            const Position& pos,
            int file
        ) {
            return
                pawnOnFile(
                    pos,
                    file,
                    false
                );
        }


        // ============================================================
        // MOBILITY + ROOK ACTIVITY
        // ============================================================

        TaperedScore evaluatePieceActivity(
            const Position& pos
        ) {
            TaperedScore score;


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


                const char type =
                    pieceType(
                        piece
                    );


                // ====================================================
                // KNIGHT MOBILITY
                // ====================================================

                if (
                    type == 'n'
                    ) {
                    const int mobility =
                        knightMobility(
                            pos,
                            square,
                            white
                        );


                    score.mg +=
                        sign *
                        mobility *
                        KNIGHT_MOBILITY_BONUS.mg;

                    score.eg +=
                        sign *
                        mobility *
                        KNIGHT_MOBILITY_BONUS.eg;


                    continue;
                }


                // ====================================================
                // BISHOP MOBILITY
                // ====================================================

                if (
                    type == 'b'
                    ) {
                    const int mobility =
                        bishopMobility(
                            pos,
                            square,
                            white
                        );


                    score.mg +=
                        sign *
                        mobility *
                        BISHOP_MOBILITY_BONUS.mg;

                    score.eg +=
                        sign *
                        mobility *
                        BISHOP_MOBILITY_BONUS.eg;


                    continue;
                }


                // ====================================================
                // ROOK
                // ====================================================

                if (
                    type == 'r'
                    ) {
                    const int mobility =
                        rookMobility(
                            pos,
                            square,
                            white
                        );


                    score.mg +=
                        sign *
                        mobility *
                        ROOK_MOBILITY_BONUS.mg;

                    score.eg +=
                        sign *
                        mobility *
                        ROOK_MOBILITY_BONUS.eg;


                    const int file =
                        fileOf(
                            square
                        );


                    const bool whitePawn =
                        fileHasWhitePawn(
                            pos,
                            file
                        );

                    const bool blackPawn =
                        fileHasBlackPawn(
                            pos,
                            file
                        );


                    const bool friendlyPawn =
                        white
                        ? whitePawn
                        : blackPawn;

                    const bool enemyPawn =
                        white
                        ? blackPawn
                        : whitePawn;


                    // ================================================
                    // OPEN FILE
                    // ================================================

                    if (
                        !friendlyPawn &&
                        !enemyPawn
                        ) {
                        score.mg +=
                            sign *
                            ROOK_OPEN_FILE_BONUS.mg;

                        score.eg +=
                            sign *
                            ROOK_OPEN_FILE_BONUS.eg;
                    }


                    // ================================================
                    // SEMI-OPEN FILE
                    // ================================================

                    else if (
                        !friendlyPawn
                        ) {
                        score.mg +=
                            sign *
                            ROOK_SEMI_OPEN_FILE_BONUS.mg;

                        score.eg +=
                            sign *
                            ROOK_SEMI_OPEN_FILE_BONUS.eg;
                    }


                    // ================================================
                    // ROOK ON THE SEVENTH
                    // ================================================

                    const int rank =
                        rankOf(
                            square
                        );


                    const bool onSeventh =
                        white
                        ? rank == 6
                        : rank == 1;


                    if (
                        onSeventh
                        ) {
                        score.mg +=
                            sign *
                            ROOK_SEVENTH_RANK_BONUS.mg;

                        score.eg +=
                            sign *
                            ROOK_SEVENTH_RANK_BONUS.eg;
                    }


                    continue;
                }


                // ====================================================
                // QUEEN MOBILITY
                // ====================================================

                if (
                    type == 'q'
                    ) {
                    const int mobility =
                        queenMobility(
                            pos,
                            square,
                            white
                        );


                    score.mg +=
                        sign *
                        mobility *
                        QUEEN_MOBILITY_BONUS.mg;

                    score.eg +=
                        sign *
                        mobility *
                        QUEEN_MOBILITY_BONUS.eg;
                }
            }


            return score;
        }
        // ============================================================
// FIND KING
// ============================================================

        int findKingSquare(
            const Position& pos,
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
                    pos.board[square] ==
                    king
                    ) {
                    return square;
                }
            }


            return -1;
        }


        // ============================================================
        // ENEMY QUEEN PRESENT
        // ============================================================

        bool enemyQueenPresent(
            const Position& pos,
            bool whiteKing
        ) {
            const char enemyQueen =
                whiteKing
                ? 'q'
                : 'Q';


            for (
                int square = 0;
                square < 64;
                ++square
                ) {
                if (
                    pos.board[square] ==
                    enemyQueen
                    ) {
                    return true;
                }
            }


            return false;
        }

        // ============================================================
// PIECE ATTACKS SQUARE
// ============================================================

        bool pieceAttacksSquare(
            const Position& pos,
            int from,
            int target
        ) {
            if (
                from < 0 ||
                from >= 64 ||
                target < 0 ||
                target >= 64 ||
                from == target
                ) {
                return false;
            }


            const char piece =
                pos.board[from];


            if (
                piece == '.'
                ) {
                return false;
            }


            const char type =
                pieceType(
                    piece
                );


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


            // ========================================================
            // KNIGHT
            // ========================================================

            if (
                type == 'n'
                ) {
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


            // ========================================================
            // BISHOP / ROOK / QUEEN
            // ========================================================

            int stepFile =
                0;

            int stepRank =
                0;


            if (
                (
                    type == 'b' ||
                    type == 'q'
                    )
                &&
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


            else if (
                (
                    type == 'r' ||
                    type == 'q'
                    )
                &&
                (
                    df == 0 ||
                    dr == 0
                    )
                ) {
                if (
                    df == 0 &&
                    dr != 0
                    ) {
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
                file != targetFile ||
                rank != targetRank
                ) {
                if (
                    file < 0 ||
                    file > 7 ||
                    rank < 0 ||
                    rank > 7
                    ) {
                    return false;
                }


                const int square =
                    rank *
                    8 +
                    file;


                if (
                    pos.board[square] != '.'
                    ) {
                    return false;
                }


                file +=
                    stepFile;

                rank +=
                    stepRank;
            }


            return true;
        }


        // ============================================================
        // KING ZONE
        // ============================================================
        //
        // King square plus the eight surrounding squares.
        //
        // ============================================================

        bool squareInKingZone(
            int square,
            int kingSquare
        ) {
            const int file =
                fileOf(
                    square
                );

            const int rank =
                rankOf(
                    square
                );

            const int kingFile =
                fileOf(
                    kingSquare
                );

            const int kingRank =
                rankOf(
                    kingSquare
                );


            return
                std::abs(
                    file -
                    kingFile
                ) <= 1
                &&
                std::abs(
                    rank -
                    kingRank
                ) <= 1;
        }


        // ============================================================
        // ATTACKER WEIGHT
        // ============================================================

        int kingAttackWeight(
            char piece
        ) {
            switch (
                pieceType(
                    piece
                )
                ) {
            case 'n':
                return
                    KING_ATTACK_KNIGHT_WEIGHT;

            case 'b':
                return
                    KING_ATTACK_BISHOP_WEIGHT;

            case 'r':
                return
                    KING_ATTACK_ROOK_WEIGHT;

            case 'q':
                return
                    KING_ATTACK_QUEEN_WEIGHT;

            default:
                return 0;
            }
        }


        // ============================================================
        // KING ATTACK PRESSURE
        // ============================================================

        TaperedScore kingAttackPressure(
            const Position& pos,
            int kingSquare,
            bool whiteKing
        ) {
            TaperedScore score;


            int attackers =
                0;

            int attackUnits =
                0;


            for (
                int from = 0;
                from < 64;
                ++from
                ) {
                const char piece =
                    pos.board[from];


                if (
                    piece == '.'
                    ) {
                    continue;
                }


                const bool pieceWhite =
                    isWhitePiece(
                        piece
                    );


                // We only want enemy pieces.
                if (
                    pieceWhite ==
                    whiteKing
                    ) {
                    continue;
                }


                const int weight =
                    kingAttackWeight(
                        piece
                    );


                if (
                    weight == 0
                    ) {
                    continue;
                }


                int zoneHits =
                    0;


                for (
                    int target = 0;
                    target < 64;
                    ++target
                    ) {
                    if (
                        !squareInKingZone(
                            target,
                            kingSquare
                        )
                        ) {
                        continue;
                    }


                    if (
                        pieceAttacksSquare(
                            pos,
                            from,
                            target
                        )
                        ) {
                        ++zoneHits;
                    }
                }


                if (
                    zoneHits > 0
                    ) {
                    ++attackers;


                    attackUnits +=
                        zoneHits *
                        weight;
                }
            }


            // ========================================================
            // MULTIPLE ATTACKERS
            // ========================================================

            if (
                attackers >= 2
                ) {
                const int extraAttackers =
                    attackers -
                    1;


                attackUnits +=
                    extraAttackers *
                    extraAttackers *
                    KING_MULTI_ATTACKER_BONUS;
            }


            // Keep this conservative until we have actual game data.
            attackUnits =
                std::min(
                    attackUnits,
                    150
                );


            score.mg =
                -attackUnits;

            score.eg =
                0;


            return score;
        }
        // ============================================================
        // KING SHIELD
        // ============================================================
        //
        // Looks at the three squares one rank in front of the king.
        //
        // Example:
        //
        // White king g1:
        //
        //     f2 g2 h2
        //
        // Black king g8:
        //
        //     f7 g7 h7
        //
        // ============================================================

        TaperedScore kingShieldScore(
            const Position& pos,
            int kingSquare,
            bool white
        ) {
            TaperedScore score;


            const int kingFile =
                fileOf(
                    kingSquare
                );

            const int kingRank =
                rankOf(
                    kingSquare
                );


            const int shieldRank =
                white
                ? kingRank + 1
                : kingRank - 1;


            if (
                shieldRank < 0 ||
                shieldRank > 7
                ) {
                return score;
            }


            const char friendlyPawn =
                white
                ? 'P'
                : 'p';


            for (
                int df = -1;
                df <= 1;
                ++df
                ) {
                const int file =
                    kingFile +
                    df;


                if (
                    file < 0 ||
                    file > 7
                    ) {
                    continue;
                }


                const int square =
                    shieldRank *
                    8 +
                    file;


                if (
                    pos.board[square] ==
                    friendlyPawn
                    ) {
                    score.mg +=
                        KING_SHIELD_PAWN_BONUS.mg;

                    score.eg +=
                        KING_SHIELD_PAWN_BONUS.eg;
                }

                else {
                    score.mg +=
                        KING_MISSING_SHIELD_PENALTY.mg;

                    score.eg +=
                        KING_MISSING_SHIELD_PENALTY.eg;
                }
            }


            return score;
        }


        // ============================================================
        // KING FILE EXPOSURE
        // ============================================================

        TaperedScore kingFileExposureScore(
            const Position& pos,
            int kingSquare,
            bool white
        ) {
            TaperedScore score;


            const int kingFile =
                fileOf(
                    kingSquare
                );


            for (
                int df = -1;
                df <= 1;
                ++df
                ) {
                const int file =
                    kingFile +
                    df;


                if (
                    file < 0 ||
                    file > 7
                    ) {
                    continue;
                }


                const bool whitePawn =
                    pawnOnFile(
                        pos,
                        file,
                        true
                    );

                const bool blackPawn =
                    pawnOnFile(
                        pos,
                        file,
                        false
                    );


                const bool friendlyPawn =
                    white
                    ? whitePawn
                    : blackPawn;

                const bool enemyPawn =
                    white
                    ? blackPawn
                    : whitePawn;


                // Completely open file.
                if (
                    !friendlyPawn &&
                    !enemyPawn
                    ) {
                    score.mg +=
                        KING_OPEN_FILE_PENALTY.mg;

                    score.eg +=
                        KING_OPEN_FILE_PENALTY.eg;
                }

                // Semi-open toward our king:
                // we have no pawn on the file.
                else if (
                    !friendlyPawn
                    ) {
                    score.mg +=
                        KING_SEMI_OPEN_FILE_PENALTY.mg;

                    score.eg +=
                        KING_SEMI_OPEN_FILE_PENALTY.eg;
                }
            }


            return score;
        }
        // ============================================================
// KING SAFETY EVALUATION
// ============================================================

        TaperedScore evaluateKingSafety(
            const Position& pos
        ) {
            TaperedScore score;


            const int whiteKing =
                findKingSquare(
                    pos,
                    true
                );

            const int blackKing =
                findKingSquare(
                    pos,
                    false
                );


            // ========================================================
            // WHITE KING
            // ========================================================

            if (
                whiteKing >= 0
                ) {
                TaperedScore whiteSafety;


                const TaperedScore shield =
                    kingShieldScore(
                        pos,
                        whiteKing,
                        true
                    );


                whiteSafety.mg +=
                    shield.mg;

                whiteSafety.eg +=
                    shield.eg;


                const TaperedScore exposure =
                    kingFileExposureScore(
                        pos,
                        whiteKing,
                        true
                    );


                whiteSafety.mg +=
                    exposure.mg;

                whiteSafety.eg +=
                    exposure.eg;

                // ====================================================
// ENEMY ATTACK PRESSURE
// ====================================================

                const TaperedScore attackPressure =
                    kingAttackPressure(
                        pos,
                        whiteKing,
                        true
                    );


                whiteSafety.mg +=
                    attackPressure.mg;

                whiteSafety.eg +=
                    attackPressure.eg;
                // Central king is more dangerous while queens remain.
                if (
                    enemyQueenPresent(
                        pos,
                        true
                    )
                    ) {
                    const int file =
                        fileOf(
                            whiteKing
                        );


                    if (
                        file >= 2 &&
                        file <= 5
                        ) {
                        whiteSafety.mg +=
                            KING_CENTER_FILE_PENALTY.mg;

                        whiteSafety.eg +=
                            KING_CENTER_FILE_PENALTY.eg;
                    }
                }


                score.mg +=
                    whiteSafety.mg;

                score.eg +=
                    whiteSafety.eg;
            }


            // ========================================================
            // BLACK KING
            // ========================================================

            if (
                blackKing >= 0
                ) {
                TaperedScore blackSafety;


                const TaperedScore shield =
                    kingShieldScore(
                        pos,
                        blackKing,
                        false
                    );


                blackSafety.mg +=
                    shield.mg;

                blackSafety.eg +=
                    shield.eg;


                const TaperedScore exposure =
                    kingFileExposureScore(
                        pos,
                        blackKing,
                        false
                    );


                blackSafety.mg +=
                    exposure.mg;

                blackSafety.eg +=
                    exposure.eg;

                // ====================================================
// ENEMY ATTACK PRESSURE
// ====================================================

                const TaperedScore attackPressure =
                    kingAttackPressure(
                        pos,
                        blackKing,
                        false
                    );


                blackSafety.mg +=
                    attackPressure.mg;

                blackSafety.eg +=
                    attackPressure.eg;
                if (
                    enemyQueenPresent(
                        pos,
                        false
                    )
                    ) {
                    const int file =
                        fileOf(
                            blackKing
                        );


                    if (
                        file >= 2 &&
                        file <= 5
                        ) {
                        blackSafety.mg +=
                            KING_CENTER_FILE_PENALTY.mg;

                        blackSafety.eg +=
                            KING_CENTER_FILE_PENALTY.eg;
                    }
                }


                // Black having good king safety is bad for White.
                score.mg -=
                    blackSafety.mg;

                score.eg -=
                    blackSafety.eg;
            }


            return score;
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

    EvaluationBreakdown evaluateDetailed(
        const Position& pos
    ) {
        EvaluationBreakdown result;


        if (
            isInsufficientMaterial(
                pos
            )
            ) {
            result.insufficientMaterial =
                true;

            result.finalScore =
                0;

            return result;
        }


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
                pos.board[square];


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


            result.materialAndPstMg +=
                sign *
                (
                    material.mg +
                    positional.mg
                    );


            result.materialAndPstEg +=
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
            result.bishopPairMg +=
                BISHOP_PAIR_BONUS.mg;

            result.bishopPairEg +=
                BISHOP_PAIR_BONUS.eg;
        }


        if (
            blackBishops >= 2
            ) {
            result.bishopPairMg -=
                BISHOP_PAIR_BONUS.mg;

            result.bishopPairEg -=
                BISHOP_PAIR_BONUS.eg;
        }


        // ========================================================
        // PAWN STRUCTURE
        // ========================================================

        const TaperedScore pawnStructure =
            evaluatePawnStructure(
                pos
            );


        result.pawnStructureMg =
            pawnStructure.mg;

        result.pawnStructureEg =
            pawnStructure.eg;


        // ========================================================
        // PIECE ACTIVITY
        // ========================================================

        const TaperedScore pieceActivity =
            evaluatePieceActivity(
                pos
            );


        result.pieceActivityMg =
            pieceActivity.mg;

        result.pieceActivityEg =
            pieceActivity.eg;


        // ========================================================
        // KING SAFETY
        // ========================================================

        const TaperedScore kingSafety =
            evaluateKingSafety(
                pos
            );


        result.kingSafetyMg =
            kingSafety.mg;

        result.kingSafetyEg =
            kingSafety.eg;


        // ========================================================
        // TEMPO
        // ========================================================

        if (
            pos.whiteToMove
            ) {
            result.tempoMg =
                TEMPO_BONUS.mg;

            result.tempoEg =
                TEMPO_BONUS.eg;
        }

        else {
            result.tempoMg =
                -TEMPO_BONUS.mg;

            result.tempoEg =
                -TEMPO_BONUS.eg;
        }


        // ========================================================
        // PHASE
        // ========================================================

        result.phase =
            std::clamp(
                phase,
                0,
                MAX_PHASE
            );


        // ========================================================
        // FINAL TOTALS
        // ========================================================

        const int totalMg =
            result.materialAndPstMg +
            result.bishopPairMg +
            result.pawnStructureMg +
            result.pieceActivityMg +
            result.kingSafetyMg +
            result.tempoMg;


        const int totalEg =
            result.materialAndPstEg +
            result.bishopPairEg +
            result.pawnStructureEg +
            result.pieceActivityEg +
            result.kingSafetyEg +
            result.tempoEg;


        result.finalScore =
            interpolate(
                totalMg,
                totalEg,
                result.phase
            );


        return result;
    }
    // ============================================================
    // FULL STATIC EVALUATION
    // ============================================================

    int evaluate(
        const Position& pos
    ) {
        return
            evaluateDetailed(
                pos
            ).finalScore;
    }

    int evaluateActive(
        const Position& pos
    ) {
        if (
            nnueEnabled() &&
            nnueLoaded()
            ) {
            return
                evaluateNNUE(
                    pos
                );
        }


        return
            evaluate(
                pos
            );
    }


    // ============================================================
    // SIDE-TO-MOVE EVALUATION
    // ============================================================

    int evaluateForSideToMove(
        const Position& pos
    ) {
        const int score =
            evaluateActive(
                pos
            );


        return
            pos.whiteToMove
            ? score
            : -score;
    }

} // namespace chess
