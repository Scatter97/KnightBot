#include "evaluation.hpp"

#include <cctype>
#include <cstdlib>

namespace chess {

    namespace {

        bool isWhitePiece(char piece) {
            return
                piece >= 'A' &&
                piece <= 'Z';
        }

        int fileOf(int square) {
            return square % 8;
        }

        int rankOf(int square) {
            return square / 8;
        }

        int positionalBonus(
            char piece,
            int square
        ) {
            const bool white =
                isWhitePiece(piece);

            const char type =
                static_cast<char>(
                    std::tolower(
                        static_cast<unsigned char>(
                            piece
                            )
                    )
                    );

            const int file =
                fileOf(square);

            const int rank =
                rankOf(square);

            const int relativeRank =
                white
                ? rank
                : 7 - rank;

            const int centerDistance =
                std::abs(file * 2 - 7) +
                std::abs(rank * 2 - 7);

            const int centerBonus =
                14 -
                centerDistance;

            switch (type) {

            case 'p':
                return
                    relativeRank * 4 +
                    centerBonus / 2;

            case 'n':
                return
                    centerBonus * 3;

            case 'b':
                return
                    centerBonus * 2;

            case 'r':
                return
                    (
                        relativeRank == 6
                        ? 18
                        : 0
                        ) +
                    centerBonus / 4;

            case 'q':
                return
                    centerBonus / 2;

            case 'k':

                if (
                    relativeRank <= 1 &&
                    (
                        file == 6 ||
                        file == 2
                        )
                    ) {
                    return 30;
                }

                return 0;

            default:
                return 0;
            }
        }


        bool hasFriendlyPawnOnFile(
            const Position& pos,
            int file,
            bool white
        ) {
            const char pawn =
                white
                ? 'P'
                : 'p';

            for (
                int rank = 0;
                rank < 8;
                ++rank
                ) {
                if (
                    pos.board[
                        rank * 8 + file
                    ] ==
                    pawn
                            ) {
                    return true;
                }
            }

            return false;
        }


        bool hasEnemyPawnOnFile(
            const Position& pos,
            int file,
            bool white
        ) {
            return
                hasFriendlyPawnOnFile(
                    pos,
                    file,
                    !white
                );
        }


        bool isPassedPawn(
            const Position& pos,
            int square,
            bool white
        ) {
            const int file =
                fileOf(square);

            const int rank =
                rankOf(square);

            const char enemyPawn =
                white
                ? 'p'
                : 'P';

            for (
                int df = -1;
                df <= 1;
                ++df
                ) {
                const int testFile =
                    file + df;

                if (
                    testFile < 0 ||
                    testFile >= 8
                    ) {
                    continue;
                }

                if (white) {

                    for (
                        int testRank =
                        rank + 1;
                        testRank < 8;
                        ++testRank
                        ) {
                        if (
                            pos.board[
                                testRank * 8 +
                                    testFile
                            ] ==
                            enemyPawn
                                    ) {
                            return false;
                        }
                    }
                }

                else {

                    for (
                        int testRank =
                        rank - 1;
                        testRank >= 0;
                        --testRank
                        ) {
                        if (
                            pos.board[
                                testRank * 8 +
                                    testFile
                            ] ==
                            enemyPawn
                                    ) {
                            return false;
                        }
                    }
                }
            }

            return true;
        }

    } // anonymous namespace


    int pieceValue(char piece) {

        const char type =
            static_cast<char>(
                std::tolower(
                    static_cast<unsigned char>(
                        piece
                        )
                )
                );

        switch (type) {

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
            return 0;

        default:
            return 0;
        }
    }


    int evaluate(
        const Position& pos
    ) {
        int score = 0;

        int whiteBishops = 0;
        int blackBishops = 0;

        int whitePawnFiles[8]{};
        int blackPawnFiles[8]{};

        // MATERIAL + BASIC POSITION
        for (
            int square = 0;
            square < 64;
            ++square
            ) {
            const char piece =
                pos.board[square];

            if (piece == '.') {
                continue;
            }

            int value =
                pieceValue(piece);

            value +=
                positionalBonus(
                    piece,
                    square
                );

            const bool white =
                isWhitePiece(piece);

            const char type =
                static_cast<char>(
                    std::tolower(
                        static_cast<unsigned char>(
                            piece
                            )
                    )
                    );

            if (white) {
                score += value;
            }

            else {
                score -= value;
            }

            if (piece == 'B') {
                ++whiteBishops;
            }

            if (piece == 'b') {
                ++blackBishops;
            }

            if (piece == 'P') {
                ++whitePawnFiles[
                    fileOf(square)
                ];
            }

            if (piece == 'p') {
                ++blackPawnFiles[
                    fileOf(square)
                ];
            }

            // PASSED PAWNS
            if (
                type == 'p' &&
                isPassedPawn(
                    pos,
                    square,
                    white
                )
                ) {
                const int relativeRank =
                    white
                    ? rankOf(square)
                    : 7 -
                    rankOf(square);

                const int bonus =
                    10 +
                    relativeRank * 8;

                score +=
                    white
                    ? bonus
                    : -bonus;
            }

            // ROOK OPEN / SEMI-OPEN FILES
            if (type == 'r') {

                const int file =
                    fileOf(square);

                const bool friendlyPawn =
                    hasFriendlyPawnOnFile(
                        pos,
                        file,
                        white
                    );

                const bool enemyPawn =
                    hasEnemyPawnOnFile(
                        pos,
                        file,
                        white
                    );

                int bonus = 0;

                if (
                    !friendlyPawn &&
                    !enemyPawn
                    ) {
                    bonus = 20;
                }

                else if (
                    !friendlyPawn
                    ) {
                    bonus = 10;
                }

                score +=
                    white
                    ? bonus
                    : -bonus;
            }
        }

        // BISHOP PAIR
        if (
            whiteBishops >= 2
            ) {
            score += 30;
        }

        if (
            blackBishops >= 2
            ) {
            score -= 30;
        }

        // DOUBLED + ISOLATED PAWNS
        for (
            int file = 0;
            file < 8;
            ++file
            ) {
            if (
                whitePawnFiles[file] > 1
                ) {
                score -=
                    (
                        whitePawnFiles[file] -
                        1
                        ) *
                    15;
            }

            if (
                blackPawnFiles[file] > 1
                ) {
                score +=
                    (
                        blackPawnFiles[file] -
                        1
                        ) *
                    15;
            }

            if (
                whitePawnFiles[file] > 0
                ) {
                const bool left =
                    file > 0 &&
                    whitePawnFiles[
                        file - 1
                    ] >
                    0;

                        const bool right =
                            file < 7 &&
                            whitePawnFiles[
                                file + 1
                            ] >
                            0;

                                if (
                                    !left &&
                                    !right
                                    ) {
                                    score -=
                                        10 *
                                        whitePawnFiles[file];
                                }
            }

            if (
                blackPawnFiles[file] > 0
                ) {
                const bool left =
                    file > 0 &&
                    blackPawnFiles[
                        file - 1
                    ] >
                    0;

                        const bool right =
                            file < 7 &&
                            blackPawnFiles[
                                file + 1
                            ] >
                            0;

                                if (
                                    !left &&
                                    !right
                                    ) {
                                    score +=
                                        10 *
                                        blackPawnFiles[file];
                                }
            }
        }

        // TEMPO / INITIATIVE BONUS
        //
        // This isn't simply "White gets points".
        // Whichever side currently has the move
        // receives a small initiative bonus.
        //
        // At the starting position this gives White
        // a small positive evaluation.
        score +=
            pos.whiteToMove
            ? 10
            : -10;

        return score;
    }


    int evaluateForSideToMove(
        const Position& pos
    ) {
        const int whiteScore =
            evaluate(pos);

        return
            pos.whiteToMove
            ? whiteScore
            : -whiteScore;
    }

} // namespace chess