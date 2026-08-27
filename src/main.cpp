#include "chess.hpp"
#include "evaluation.hpp"
#include "search.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Fallback in case KnightBot is ever compiled without CMake.
#ifndef KNIGHTBOT_VERSION
#define KNIGHTBOT_VERSION "development"
#endif

namespace {

    using chess::Move;
    using chess::Position;
    using chess::SearchResult;


    // ============================================================
    // PLAY UCI MOVE
    // ============================================================

    bool playUciMove(
        Position& position,
        const std::string& uci
    ) {
        const auto moves =
            chess::generateLegalMoves(
                position
            );

        for (
            const Move& move :
            moves
            ) {
            if (
                chess::moveToUci(move) ==
                uci
                ) {
                chess::makeMove(
                    position,
                    move
                );

                return true;
            }
        }

        return false;
    }


    // ============================================================
    // WHITE-PERSPECTIVE SCORE
    // ============================================================

    int whitePerspectiveScore(
        const Position& root,
        int searchScore
    ) {
        return
            root.whiteToMove
            ? searchScore
            : -searchScore;
    }


    // ============================================================
    // HUMAN SCORE DISPLAY
    // ============================================================

    void printHumanScore(
        const Position& root,
        int searchScore
    ) {
        const int score =
            whitePerspectiveScore(
                root,
                searchScore
            );


        if (
            std::abs(score) >=
            chess::MATE_THRESHOLD
            ) {
            const int plies =
                chess::MATE_SCORE -
                std::abs(score);

            const int moves =
                (plies + 1) /
                2;


            if (
                score > 0
                ) {
                std::cout <<
                    "Evaluation: White mates in " <<
                    moves <<
                    '\n';
            }

            else {
                std::cout <<
                    "Evaluation: Black mates in " <<
                    moves <<
                    '\n';
            }

            return;
        }


        std::cout <<
            std::fixed <<
            std::setprecision(2) <<
            "Evaluation: " <<
            (
                static_cast<double>(
                    score
                    ) /
                100.0
                );


        if (
            score > 0
            ) {
            std::cout <<
                " (White better)";
        }

        else if (
            score < 0
            ) {
            std::cout <<
                " (Black better)";
        }

        else {
            std::cout <<
                " (Equal)";
        }


        std::cout <<
            '\n';
    }


    // ============================================================
    // SAN PRINCIPAL VARIATION
    // ============================================================

    void printSanPV(
        Position position,
        const std::vector<Move>& pv
    ) {
        if (
            pv.empty()
            ) {
            return;
        }


        std::cout <<
            "PV:    ";


        for (
            const Move& move :
            pv
            ) {
            std::cout <<
                chess::moveToSan(
                    position,
                    move
                )
                <<
                ' ';


            chess::makeMove(
                position,
                move
            );
        }


        std::cout <<
            '\n';
    }


    // ============================================================
    // HUMAN SEARCH OUTPUT
    // ============================================================

    void printHumanSearchResult(
        const Position& root,
        const SearchResult& result
    ) {
        if (
            !result.hasMove
            ) {
            if (
                result.score <=
                -chess::MATE_THRESHOLD
                ) {
                std::cout <<
                    "Checkmate.\n";
            }

            else {
                std::cout <<
                    "Stalemate.\n";
            }

            return;
        }


        std::cout <<
            "Best move: " <<
            chess::moveToSan(
                root,
                result.bestMove
            )
            <<
            '\n';


        printHumanScore(
            root,
            result.score
        );


        std::cout <<
            "Depth: " <<
            result.depth <<
            '\n';


        std::cout <<
            "Nodes: " <<
            result.nodes <<
            '\n';


        std::cout <<
            std::fixed <<
            std::setprecision(4) <<
            "Time:  " <<
            result.seconds <<
            " seconds\n";


        if (
            result.seconds > 0.0
            ) {
            const auto nps =
                static_cast<std::uint64_t>(
                    result.nodes /
                    result.seconds
                    );


            std::cout <<
                "NPS:   " <<
                nps <<
                '\n';
        }


        std::cout <<
            "TT entries: " <<
            chess::transpositionTableSize() <<
            '\n';


        printSanPV(
            root,
            result.pv
        );
    }


    // ============================================================
    // UCI OUTPUT
    // ============================================================

    void printUciInfo(
        const SearchResult& result
    ) {
        std::cout <<
            "info depth " <<
            result.depth;


        if (
            std::abs(result.score) >=
            chess::MATE_THRESHOLD
            ) {
            const int plies =
                chess::MATE_SCORE -
                std::abs(result.score);

            const int moves =
                (plies + 1) /
                2;


            std::cout <<
                " score mate " <<
                (
                    result.score > 0
                    ? moves
                    : -moves
                    );
        }

        else {
            std::cout <<
                " score cp " <<
                result.score;
        }


        std::cout <<
            " nodes " <<
            result.nodes;


        if (
            result.seconds > 0.0
            ) {
            const auto nps =
                static_cast<std::uint64_t>(
                    result.nodes /
                    result.seconds
                    );


            std::cout <<
                " nps " <<
                nps;


            std::cout <<
                " time " <<
                static_cast<std::uint64_t>(
                    result.seconds *
                    1000.0
                    );
        }


        if (
            !result.pv.empty()
            ) {
            std::cout <<
                " pv";


            for (
                const Move& move :
                result.pv
                ) {
                std::cout <<
                    ' ' <<
                    chess::moveToUci(
                        move
                    );
            }
        }


        std::cout <<
            '\n';


        if (
            result.hasMove
            ) {
            std::cout <<
                "bestmove " <<
                chess::moveToUci(
                    result.bestMove
                )
                <<
                '\n';
        }

        else {
            std::cout <<
                "bestmove 0000\n";
        }
    }


    // ============================================================
    // HELP
    // ============================================================

    void printHelp() {
        std::cout <<
            "\nKnightBot CLI commands:\n"
            "  board\n"
            "  moves\n"
            "  move <uci>\n"
            "  eval\n"
            "  bestmove <depth>\n"
            "  think <milliseconds>\n"
            "  playbest <depth>\n"
            "  playtime <milliseconds>\n"
            "  perft <depth>\n"
            "  divide <depth>\n"
            "  startpos\n"
            "  fen <FEN string>\n"
            "  showfen\n"
            "  ttclear\n"
            "  ttstats\n"
            "  help\n"
            "  quit\n\n"

            "UCI commands:\n"
            "  uci\n"
            "  isready\n"
            "  ucinewgame\n"
            "  position startpos\n"
            "  position startpos moves e2e4 e7e5\n"
            "  position fen <FEN>\n"
            "  go depth <n>\n"
            "  go movetime <ms>\n\n";
    }


    // ============================================================
    // UCI POSITION
    // ============================================================

    bool handleUciPosition(
        const std::string& line,
        Position& position
    ) {
        std::istringstream input(
            line
        );

        std::string command;
        std::string token;


        input >>
            command;


        if (
            command !=
            "position"
            ) {
            return false;
        }


        input >>
            token;


        // ========================================================
        // STARTPOS
        // ========================================================

        if (
            token ==
            "startpos"
            ) {
            position =
                chess::startPosition();


            if (
                input >>
                token
                ) {
                if (
                    token ==
                    "moves"
                    ) {
                    while (
                        input >>
                        token
                        ) {
                        if (
                            !playUciMove(
                                position,
                                token
                            )
                            ) {
                            return false;
                        }
                    }
                }
            }


            return true;
        }


        // ========================================================
        // FEN
        // ========================================================

        if (
            token ==
            "fen"
            ) {
            std::vector<std::string>
                fenFields;


            for (
                int i = 0;
                i < 6;
                ++i
                ) {
                if (
                    !(input >> token)
                    ) {
                    return false;
                }

                fenFields.push_back(
                    token
                );
            }


            std::string fen;


            for (
                std::size_t i = 0;
                i < fenFields.size();
                ++i
                ) {
                if (
                    i != 0
                    ) {
                    fen +=
                        ' ';
                }

                fen +=
                    fenFields[i];
            }


            try {
                position =
                    chess::fromFEN(
                        fen
                    );
            }

            catch (...) {
                return false;
            }


            if (
                input >>
                token
                ) {
                if (
                    token ==
                    "moves"
                    ) {
                    while (
                        input >>
                        token
                        ) {
                        if (
                            !playUciMove(
                                position,
                                token
                            )
                            ) {
                            return false;
                        }
                    }
                }
            }


            return true;
        }


        return false;
    }


    // ============================================================
    // UCI GO
    // ============================================================

    bool handleUciGo(
        const std::string& line,
        const Position& position
    ) {
        std::istringstream input(
            line
        );

        std::string command;
        std::string token;


        input >>
            command;


        if (
            command !=
            "go"
            ) {
            return false;
        }


        int depth =
            -1;

        int movetime =
            -1;


        while (
            input >>
            token
            ) {
            if (
                token ==
                "depth"
                ) {
                input >>
                    depth;
            }

            else if (
                token ==
                "movetime"
                ) {
                input >>
                    movetime;
            }
        }


        SearchResult result;


        if (
            movetime > 0
            ) {
            result =
                chess::searchBestMoveTimed(
                    position,
                    movetime
                );
        }

        else {
            if (
                depth < 1
                ) {
                depth =
                    5;
            }


            result =
                chess::searchBestMove(
                    position,
                    depth
                );
        }


        printUciInfo(
            result
        );


        return true;
    }

} // anonymous namespace


int main() {

    Position position =
        chess::startPosition();


    bool uciMode =
        false;


    // ========================================================
    // STARTUP BANNER
    // ========================================================

    std::cout <<
        "=========================\n"
        "     KnightBot v" <<
        KNIGHTBOT_VERSION <<
        "\n"
        "=========================\n\n"
        "Incremental bitboard alpha-beta engine\n";


    printHelp();


    chess::printBoard(
        position
    );


    std::string line;


    while (
        true
        ) {
        if (
            !uciMode
            ) {
            std::cout <<
                "\n> ";
        }


        if (
            !std::getline(
                std::cin,
                line
            )
            ) {
            break;
        }


        std::istringstream input(
            line
        );


        std::string command;

        input >>
            command;


        if (
            command.empty()
            ) {
            continue;
        }


        // ====================================================
        // QUIT
        // ====================================================

        if (
            command == "quit" ||
            command == "exit"
            ) {
            break;
        }


        // ====================================================
        // UCI
        // ====================================================

        if (
            command ==
            "uci"
            ) {
            uciMode =
                true;


            std::cout <<
                "id name KnightBot " <<
                KNIGHTBOT_VERSION <<
                '\n';


            std::cout <<
                "id author Joshua Wang\n";


            std::cout <<
                "uciok\n";


            continue;
        }


        if (
            command ==
            "isready"
            ) {
            std::cout <<
                "readyok\n";

            continue;
        }


        if (
            command ==
            "ucinewgame"
            ) {
            position =
                chess::startPosition();


            chess::clearTranspositionTable();


            continue;
        }


        if (
            command ==
            "position"
            ) {
            handleUciPosition(
                line,
                position
            );

            continue;
        }


        if (
            command ==
            "go"
            ) {
            handleUciGo(
                line,
                position
            );

            continue;
        }


        // ====================================================
        // HELP
        // ====================================================

        if (
            command ==
            "help"
            ) {
            printHelp();
        }


        // ====================================================
        // BOARD
        // ====================================================

        else if (
            command ==
            "board"
            ) {
            chess::printBoard(
                position
            );
        }


        // ====================================================
        // MOVES
        // ====================================================

        else if (
            command ==
            "moves"
            ) {
            const auto moves =
                chess::generateLegalMoves(
                    position
                );


            std::cout <<
                "Legal moves (" <<
                moves.size() <<
                "):\n";


            for (
                const Move& move :
                moves
                ) {
                std::cout <<
                    chess::moveToSan(
                        position,
                        move
                    )
                    <<
                    ' ';
            }


            std::cout <<
                '\n';
        }


        // ====================================================
        // HUMAN MOVE
        // ====================================================

        else if (
            command ==
            "move"
            ) {
            std::string uci;

            input >>
                uci;


            if (
                uci.empty()
                ) {
                std::cout <<
                    "Usage: move e2e4\n";

                continue;
            }


            const auto legalMoves =
                chess::generateLegalMoves(
                    position
                );


            bool found =
                false;


            for (
                const Move& move :
                legalMoves
                ) {
                if (
                    chess::moveToUci(
                        move
                    ) ==
                    uci
                    ) {
                    const std::string san =
                        chess::moveToSan(
                            position,
                            move
                        );


                    chess::makeMove(
                        position,
                        move
                    );


                    std::cout <<
                        "Played " <<
                        san <<
                        '\n';


                    found =
                        true;


                    break;
                }
            }


            if (
                !found
                ) {
                std::cout <<
                    "Illegal move: " <<
                    uci <<
                    '\n';
            }

            else {
                chess::printBoard(
                    position
                );
            }
        }


        // ====================================================
        // STATIC EVAL
        // ====================================================

        else if (
            command ==
            "eval"
            ) {
            const int score =
                chess::evaluate(
                    position
                );


            std::cout <<
                std::fixed <<
                std::setprecision(2) <<
                "Evaluation: " <<
                (
                    static_cast<double>(
                        score
                        )
                    /
                    100.0
                    );


            if (
                score > 0
                ) {
                std::cout <<
                    " (White better)";
            }

            else if (
                score < 0
                ) {
                std::cout <<
                    " (Black better)";
            }

            else {
                std::cout <<
                    " (Equal)";
            }


            std::cout <<
                '\n';
        }


        // ====================================================
        // FIXED DEPTH SEARCH
        // ====================================================

        else if (
            command ==
            "bestmove"
            ) {
            int depth =
                0;


            if (
                !(input >> depth)
                ||
                depth < 1
                ) {
                std::cout <<
                    "Usage: bestmove <depth>\n";

                continue;
            }


            std::cout <<
                "Iterative search to depth " <<
                depth <<
                "...\n";


            const SearchResult result =
                chess::searchBestMove(
                    position,
                    depth
                );


            printHumanSearchResult(
                position,
                result
            );
        }


        // ====================================================
        // TIMED SEARCH
        // ====================================================

        else if (
            command ==
            "think"
            ) {
            int milliseconds =
                0;


            if (
                !(input >> milliseconds)
                ||
                milliseconds < 1
                ) {
                std::cout <<
                    "Usage: think <milliseconds>\n";

                continue;
            }


            std::cout <<
                "KnightBot thinking for about " <<
                milliseconds <<
                " ms...\n";


            const SearchResult result =
                chess::searchBestMoveTimed(
                    position,
                    milliseconds
                );


            printHumanSearchResult(
                position,
                result
            );
        }


        // ====================================================
        // PLAY BEST
        // ====================================================

        else if (
            command ==
            "playbest"
            ) {
            int depth =
                0;


            if (
                !(input >> depth)
                ||
                depth < 1
                ) {
                std::cout <<
                    "Usage: playbest <depth>\n";

                continue;
            }


            const Position root =
                position;


            const SearchResult result =
                chess::searchBestMove(
                    position,
                    depth
                );


            printHumanSearchResult(
                root,
                result
            );


            if (
                result.hasMove
                ) {
                chess::makeMove(
                    position,
                    result.bestMove
                );


                chess::printBoard(
                    position
                );
            }
        }


        // ====================================================
        // PLAY TIMED
        // ====================================================

        else if (
            command ==
            "playtime"
            ) {
            int milliseconds =
                0;


            if (
                !(input >> milliseconds)
                ||
                milliseconds < 1
                ) {
                std::cout <<
                    "Usage: playtime <milliseconds>\n";

                continue;
            }


            const Position root =
                position;


            const SearchResult result =
                chess::searchBestMoveTimed(
                    position,
                    milliseconds
                );


            printHumanSearchResult(
                root,
                result
            );


            if (
                result.hasMove
                ) {
                chess::makeMove(
                    position,
                    result.bestMove
                );


                chess::printBoard(
                    position
                );
            }
        }


        // ====================================================
        // PERFT
        // ====================================================

        else if (
            command ==
            "perft"
            ) {
            int depth =
                0;


            if (
                !(input >> depth)
                ||
                depth < 0
                ) {
                std::cout <<
                    "Usage: perft <depth>\n";

                continue;
            }


            const auto start =
                std::chrono::steady_clock::now();


            const std::uint64_t nodes =
                chess::perft(
                    position,
                    depth
                );


            const auto end =
                std::chrono::steady_clock::now();


            const std::chrono::duration<double>
                elapsed =
                end -
                start;


            std::cout <<
                "Depth: " <<
                depth <<
                '\n';


            std::cout <<
                "Nodes: " <<
                nodes <<
                '\n';


            std::cout <<
                std::fixed <<
                std::setprecision(6) <<
                "Time:  " <<
                elapsed.count() <<
                " seconds\n";


            if (
                elapsed.count() >
                0.0
                ) {
                const auto nps =
                    static_cast<std::uint64_t>(
                        nodes /
                        elapsed.count()
                        );


                std::cout <<
                    "NPS:   " <<
                    nps <<
                    '\n';
            }
        }


        // ====================================================
        // DIVIDE
        // ====================================================

        else if (
            command ==
            "divide"
            ) {
            int depth =
                0;


            if (
                !(input >> depth)
                ||
                depth < 1
                ) {
                std::cout <<
                    "Usage: divide <depth>\n";

                continue;
            }


            const auto moves =
                chess::generateLegalMoves(
                    position
                );


            std::uint64_t total =
                0;


            for (
                const Move& move :
                moves
                ) {
                Position next =
                    position;


                chess::makeMove(
                    next,
                    move
                );


                const std::uint64_t count =
                    chess::perft(
                        next,
                        depth - 1
                    );


                total +=
                    count;


                std::cout <<
                    chess::moveToSan(
                        position,
                        move
                    )
                    <<
                    ": " <<
                    count <<
                    '\n';
            }


            std::cout <<
                "Total: " <<
                total <<
                '\n';
        }


        // ====================================================
        // START POSITION
        // ====================================================

        else if (
            command ==
            "startpos"
            ) {
            position =
                chess::startPosition();


            chess::clearTranspositionTable();


            std::cout <<
                "Loaded starting position.\n";


            chess::printBoard(
                position
            );
        }


        // ====================================================
        // FEN
        // ====================================================

        else if (
            command ==
            "fen"
            ) {
            std::string fen;


            std::getline(
                input,
                fen
            );


            if (
                !fen.empty()
                &&
                fen.front() ==
                ' '
                ) {
                fen.erase(
                    fen.begin()
                );
            }


            try {
                position =
                    chess::fromFEN(
                        fen
                    );


                chess::clearTranspositionTable();


                std::cout <<
                    "FEN loaded.\n";


                chess::printBoard(
                    position
                );
            }

            catch (
                const std::exception& error
                ) {
                std::cout <<
                    "FEN error: " <<
                    error.what() <<
                    '\n';
            }
        }


        // ====================================================
        // SHOW FEN
        // ====================================================

        else if (
            command ==
            "showfen"
            ) {
            std::cout <<
                chess::toFEN(
                    position
                )
                <<
                '\n';
        }


        // ====================================================
        // TRANSPOSITION TABLE
        // ====================================================

        else if (
            command ==
            "ttclear"
            ) {
            chess::clearTranspositionTable();


            std::cout <<
                "Transposition table cleared.\n";
        }


        else if (
            command ==
            "ttstats"
            ) {
            std::cout <<
                "TT entries: " <<
                chess::transpositionTableSize() <<
                '\n';
        }


        // ====================================================
        // UNKNOWN
        // ====================================================

        else {
            if (
                !uciMode
                ) {
                std::cout <<
                    "Unknown command: " <<
                    command <<
                    '\n';


                printHelp();
            }
        }
    }


    if (
        !uciMode
        ) {
        std::cout <<
            "\nKnightBot stopped.\n";
    }


    return 0;
}