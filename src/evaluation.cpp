#include "evaluation.hpp"

#include <algorithm>
#include <array>
#include <cctype>

namespace chess {

    namespace {

        // ============================================================
        // TAPERED EVALUATION
        // ============================================================
        //
        // KnightBot now maintains TWO scores:
        //
        //     middlegame score
        //     endgame score
        //
        // Then blends between them according to how much material
        // remains.
        //
        // Example:
        //
        // Opening:
        //     mostly middlegame evaluation
        //
        // Queenless simplified position:
        //     mixture
        //
        // King + pawn ending:
        //     mostly endgame evaluation
        //
        // ============================================================

        struct TaperedScore {
            int mg = 0;
            int eg = 0;
        };


        // ============================================================
        // GAME PHASE
        // ============================================================
        //
        // Typical phase weights:
        //
        // Knight = 1
        // Bishop = 1
        // Rook   = 2
        // Queen  = 4
        //
        // With both armies:
        //
        // 4 knights = 4
        // 4 bishops = 4
        // 4 rooks   = 8
        // 2 queens  = 8
        //
        // Total = 24
        //
        // ============================================================

        constexpr int MAX_PHASE = 24;


        // ============================================================
        // MATERIAL VALUES
        // ============================================================
        //
        // Separate middlegame and endgame values.
        //
        // Pawns become slightly more important as the board empties.
        // Rooks also improve somewhat.
        //
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
        //
        // Tables are written from White's point of view:
        //
        // index 0  = a1
        // index 7  = h1
        // index 56 = a8
        // index 63 = h8
        //
        // Black uses a vertically mirrored square.
        //
        // These are deliberately moderate. We don't want the PSTs
        // overpowering actual material.
        //
        // ============================================================


        // ------------------------------------------------------------
        // PAWN - MIDDLEGAME
        // ------------------------------------------------------------

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


        // ------------------------------------------------------------
        // PAWN - ENDGAME
        // ------------------------------------------------------------

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


        // ------------------------------------------------------------
        // KNIGHT - MIDDLEGAME
        // ------------------------------------------------------------

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


        // ------------------------------------------------------------
        // KNIGHT - ENDGAME
        // ------------------------------------------------------------

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


        // ------------------------------------------------------------
        // BISHOP - MIDDLEGAME
        // ------------------------------------------------------------

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


        // ------------------------------------------------------------
        // BISHOP - ENDGAME
        // ------------------------------------------------------------

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


        // ------------------------------------------------------------
        // ROOK - MIDDLEGAME
        // ------------------------------------------------------------

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


        // ------------------------------------------------------------
        // ROOK - ENDGAME
        // ------------------------------------------------------------

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


        // ------------------------------------------------------------
        // QUEEN - MIDDLEGAME
        // ------------------------------------------------------------

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


        // ------------------------------------------------------------
        // QUEEN - ENDGAME
        // ------------------------------------------------------------

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


        // ------------------------------------------------------------
        // KING - MIDDLEGAME
        // ------------------------------------------------------------
        //
        // King prefers safety near its home rank.
        //
        // ------------------------------------------------------------

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


        // ------------------------------------------------------------
        // KING - ENDGAME
        // ------------------------------------------------------------
        //
        // In the endgame, central king activity is valuable.
        //
        // ------------------------------------------------------------

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
        // SQUARE HELPERS
        // ============================================================

        int mirrorSquare(
            int square
        ) {
            // Flip rank, preserve file.
            //
            // a1 <-> a8
            // e2 <-> e7

            return
                square ^
                56;
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
        // PIECE TYPE
        // ============================================================

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


        // ============================================================
        // PIECE MATERIAL
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
        // PIECE-SQUARE BONUS
        // ============================================================

        TaperedScore pieceSquareScore(
            char piece,
            int square
        ) {
            const bool white =
                piece >= 'A' &&
                piece <= 'Z';


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
        // PHASE WEIGHT
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
        // BISHOP PAIR
        // ============================================================
        //
        // Small tapered bishop-pair bonus.
        //
        // ============================================================

        constexpr TaperedScore BISHOP_PAIR_BONUS = {
            30,
            45
        };


        // ============================================================
        // TEMPO
        // ============================================================
        //
        // A small bonus for having the move.
        //
        // This is applied after converting to White's perspective.
        //
        // ============================================================

        constexpr TaperedScore TEMPO_BONUS = {
            10,
            6
        };


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
                pos.board[square];


            if (
                piece == '.'
                ) {
                continue;
            }


            const bool white =
                piece >= 'A' &&
                piece <= 'Z';


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
        // TAPER
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