#include "chess.hpp"
#include "devtools.hpp"
#include "evaluation.hpp"
#include "nnue.hpp"
#include "search.hpp"
#include "training.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif


// Fallback in case KnightBot is ever compiled without CMake.
#ifndef KNIGHTBOT_VERSION
#define KNIGHTBOT_VERSION "development"
#endif

namespace {

    using chess::Move;
    using chess::Position;
    using chess::SearchResult;

    std::mutex uciOutputMutex;
    std::thread uciSearchThread;

    void stopAndJoinUciSearch() {
        if (uciSearchThread.joinable()) {
            chess::requestSearchStop();
            uciSearchThread.join();
        }

        chess::resetSearchStop();
    }


    std::string trimCopy(
        std::string text
    ) {
        while (!text.empty() &&
               std::isspace(static_cast<unsigned char>(text.front()))) {
            text.erase(text.begin());
        }

        while (!text.empty() &&
               std::isspace(static_cast<unsigned char>(text.back()))) {
            text.pop_back();
        }

        return text;
    }


    std::string lowerCopy(
        std::string text
    ) {
        std::transform(
            text.begin(),
            text.end(),
            text.begin(),
            [](unsigned char character) {
                return static_cast<char>(std::tolower(character));
            }
        );

        return text;
    }


    bool handleUciSetOption(
        const std::string& line
    ) {
        constexpr const char* prefix = "setoption name ";

        if (line.rfind(prefix, 0) != 0) {
            return false;
        }

        std::string remainder =
            line.substr(std::char_traits<char>::length(prefix));
        std::string name;
        std::string value;

        const std::size_t valuePosition = remainder.find(" value ");

        if (valuePosition == std::string::npos) {
            name = trimCopy(remainder);
        }
        else {
            name = trimCopy(remainder.substr(0, valuePosition));
            value = trimCopy(remainder.substr(valuePosition + 7));
        }

        if (name == "UseNNUE") {
            const std::string lowered = lowerCopy(value);
            const bool enabled =
                lowered == "true" || lowered == "1" ||
                lowered == "on" || lowered == "yes";

            chess::setNNUEEnabled(enabled);
            chess::clearTranspositionTable();

            std::cout << "info string UseNNUE "
                      << (enabled ? "true" : "false");

            if (enabled && !chess::nnueLoaded()) {
                std::cout << " - no network loaded; using handcrafted fallback";
            }

            std::cout << '\n' << std::flush;
            return true;
        }

        if (name == "EvalFile") {
            if (value.empty() || value == "<empty>") {
                chess::unloadNNUE();
                chess::clearTranspositionTable();
                std::cout << "info string NNUE network unloaded\n"
                          << std::flush;
                return true;
            }

            std::string error;

            if (chess::loadNNUE(value, &error)) {
                chess::clearTranspositionTable();
                std::cout << "info string NNUE loaded: " << value << '\n'
                          << std::flush;
            }
            else {
                std::cout << "info string NNUE load failed: " << error << '\n'
                          << std::flush;
            }

            return true;
        }

        std::cout << "info string Unknown option: " << name << '\n'
                  << std::flush;
        return true;
    }


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

    // ============================================================
// UCI SEARCH INFO
// ============================================================

    void printUciSearchInfo(
        const SearchResult& result
    ) {
        const std::lock_guard<std::mutex> lock(uciOutputMutex);
        std::cout <<
            "info depth " <<
            result.depth;


        // ========================================================
        // SCORE
        // ========================================================

        if (
            std::abs(result.score) >=
            chess::MATE_THRESHOLD
            ) {
            const int plies =
                chess::MATE_SCORE -
                std::abs(
                    result.score
                );


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


        // ========================================================
        // NODES
        // ========================================================

        std::cout <<
            " nodes " <<
            result.nodes;


        // ========================================================
        // TIME + NPS
        // ========================================================

        if (
            result.seconds > 0.0
            ) {
            const auto milliseconds =
                static_cast<std::uint64_t>(
                    result.seconds *
                    1000.0
                    );


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
                milliseconds;
        }


        // ========================================================
        // PRINCIPAL VARIATION
        // ========================================================

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


        // Make sure GUIs receive the line immediately.
        std::cout.flush();
    }


    // ============================================================
    // UCI BEST MOVE
    // ============================================================

    void printUciBestMove(
        const SearchResult& result
    ) {
        const std::lock_guard<std::mutex> lock(uciOutputMutex);
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


        std::cout.flush();
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
            "  evaldetail\n"
            "  evalcompare\n"
            "  nnue on|off\n"
            "  nnueload <file>\n"
            "  nnueinfo\n"
            "  nnuegentest <file>\n"
            "  exportpos <file> [target_cp]\n"
            "  selftest\n"
            "  bench\n"
            "  benchfull\n"
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
            "  setoption name UseNNUE value true|false\n"
            "  setoption name EvalFile value <path>\n"
            "  go depth <n>\n"
            "  go movetime <ms>\n\n";
    }


    // ============================================================
    // UCI POSITION
    // ============================================================

    bool handleUciPosition(
        const std::string& line,
        Position& position,
        chess::SearchHistory& history
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

            history = {position.zobristKey};


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

                        history.push_back(position.zobristKey);
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

                history = {position.zobristKey};
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

                        history.push_back(position.zobristKey);
                    }
                }
            }


            return true;
        }


        return false;
    }
    // ============================================================
// UCI TIME MANAGEMENT
// ============================================================
//
// Calculates how many milliseconds KnightBot should spend on
// the current move.
//
// The goal is to:
//
// - avoid running out of time
// - use more time when plenty remains
// - take the increment into account
// - respect "movestogo" when a GUI supplies it
//
// This is deliberately conservative for now. We can make the
// time manager smarter later.
//
// ============================================================

    // ============================================================
// UCI TIME MANAGEMENT
// ============================================================
//
// KnightBot time manager.
//
// Goals:
//
// - remain conservative in the opening
// - spend more time in the middlegame
// - use increment intelligently
// - avoid dying on the clock
// - preserve a safety reserve
// - become increasingly careful in severe time pressure
//
// This returns the target number of milliseconds for the
// current search.
//
// ============================================================

    int calculateMoveTime(
        const Position& position,
        int whiteTime,
        int blackTime,
        int whiteIncrement,
        int blackIncrement,
        int movesToGo
    ) {
        const int remainingTime =
            position.whiteToMove
            ? whiteTime
            : blackTime;


        const int increment =
            position.whiteToMove
            ? whiteIncrement
            : blackIncrement;


        // No usable clock information.
        if (
            remainingTime < 0
            ) {
            return -1;
        }


        // ========================================================
        // SAFETY RESERVE
        // ========================================================
        //
        // Don't plan to consume the entire clock.
        //
        // Example:
        //
        // 10,000 ms -> ~833 ms reserve
        //  4,000 ms -> ~333 ms reserve
        //  1,000 ms -> ~83 ms reserve
        //
        // ========================================================

        const int reserve =
            std::clamp(
                remainingTime / 12,
                75,
                1200
            );


        const int usableTime =
            std::max(
                1,
                remainingTime -
                reserve
            );


        // ========================================================
        // ESTIMATED MOVES REMAINING
        // ========================================================
        //
        // If the GUI explicitly supplies movestogo, use it.
        //
        // Otherwise estimate based on where we are in the game.
        //
        // Opening:
        //     save some time
        //
        // Middlegame:
        //     spend substantially more
        //
        // Late game:
        //     fewer moves probably remain
        //
        // ========================================================

        int expectedMoves;


        if (
            movesToGo > 0
            ) {
            expectedMoves =
                std::clamp(
                    movesToGo,
                    1,
                    60
                );
        }

        else {
            const int moveNumber =
                position.fullmoveNumber;


            if (
                moveNumber <= 10
                ) {
                expectedMoves =
                    24;
            }

            else if (
                moveNumber <= 25
                ) {
                expectedMoves =
                    14;
            }

            else if (
                moveNumber <= 40
                ) {
                expectedMoves =
                    12;
            }

            else {
                expectedMoves =
                    10;
            }
        }


        // ========================================================
        // BASIC TIME ALLOCATION
        // ========================================================

        int moveTime =
            usableTime /
            expectedMoves;


        // ========================================================
        // INCREMENT
        // ========================================================
        //
        // Spend most, but not all, of the increment.
        //
        // At +0.1:
        //
        // 100 ms increment -> +80 ms
        //
        // ========================================================

        if (
            increment > 0
            ) {
            moveTime +=
                (
                    increment *
                    4
                    )
                /
                5;
        }


        // ========================================================
        // GAME-STAGE MULTIPLIER
        // ========================================================
        //
        // Opening:
        //     85%
        //
        // Early/mid middlegame:
        //     135%
        //
        // Later middlegame:
        //     120%
        //
        // Endgame:
        //     105%
        //
        // This is specifically intended to prevent KnightBot from
        // saving several seconds unnecessarily through the middle
        // of a 10+0.1 game.
        //
        // ========================================================

        const int moveNumber =
            position.fullmoveNumber;


        if (
            moveNumber <= 10
            ) {
            moveTime =
                (
                    moveTime *
                    85
                    )
                /
                100;
        }

        else if (
            moveNumber <= 25
            ) {
            moveTime =
                (
                    moveTime *
                    135
                    )
                /
                100;
        }

        else if (
            moveNumber <= 40
            ) {
            moveTime =
                (
                    moveTime *
                    120
                    )
                /
                100;
        }

        else {
            moveTime =
                (
                    moveTime *
                    105
                    )
                /
                100;
        }


        // ========================================================
        // HARD MAXIMUM
        // ========================================================
        //
        // Never intentionally spend more than about one third of
        // our usable clock on a normal move.
        //
        // ========================================================

        int maximumMoveTime =
            std::max(
                1,
                usableTime /
                3
            );


        // ========================================================
        // LOW-TIME EMERGENCY MODE
        // ========================================================

        if (
            remainingTime <= 1000
            ) {
            maximumMoveTime =
                std::min(
                    maximumMoveTime,
                    std::max(
                        1,
                        remainingTime /
                        6
                    )
                );
        }


        if (
            remainingTime <= 400
            ) {
            maximumMoveTime =
                std::min(
                    maximumMoveTime,
                    std::max(
                        1,
                        remainingTime /
                        10
                    )
                );
        }


        if (
            remainingTime <= 150
            ) {
            maximumMoveTime =
                std::min(
                    maximumMoveTime,
                    std::max(
                        1,
                        remainingTime /
                        15
                    )
                );
        }


        moveTime =
            std::min(
                moveTime,
                maximumMoveTime
            );


        return
            std::max(
                1,
                moveTime
            );
    }
    // ============================================================
// UCI GO
// ============================================================

    bool handleUciGo(
        const std::string& line,
        const Position& position,
        const chess::SearchHistory& history
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


        // ========================================================
        // UCI SEARCH PARAMETERS
        // ========================================================

        int depth =
            -1;

        int movetime =
            -1;

        int whiteTime =
            -1;

        int blackTime =
            -1;

        int whiteIncrement =
            0;

        int blackIncrement =
            0;

        int movesToGo =
            -1;

        bool infinite =
            false;

        auto readInteger = [&input](int& value) {
            std::string text;
            if (!(input >> text)) return false;
            try {
                std::size_t consumed = 0;
                const long long parsed = std::stoll(text, &consumed);
                if (consumed != text.size() || parsed < 0 ||
                    parsed > std::numeric_limits<int>::max()) {
                    return false;
                }
                value = static_cast<int>(parsed);
                return true;
            }
            catch (...) {
                return false;
            }
        };


        // ========================================================
        // PARSE GO COMMAND
        // ========================================================

        while (
            input >>
            token
            ) {
            if (
                token ==
                "depth"
                ) {
                if (!readInteger(depth)) return false;
            }

            else if (
                token ==
                "movetime"
                ) {
                if (!readInteger(movetime)) return false;
            }

            else if (
                token ==
                "wtime"
                ) {
                if (!readInteger(whiteTime)) return false;
            }

            else if (
                token ==
                "btime"
                ) {
                if (!readInteger(blackTime)) return false;
            }

            else if (
                token ==
                "winc"
                ) {
                if (!readInteger(whiteIncrement)) return false;
            }

            else if (
                token ==
                "binc"
                ) {
                if (!readInteger(blackIncrement)) return false;
            }

            else if (
                token ==
                "movestogo"
                ) {
                if (!readInteger(movesToGo)) return false;
            }

            else if (
                token ==
                "infinite"
                ) {
                infinite =
                    true;
            }
        }


        // ========================================================
        // LIVE ITERATIVE-DEEPENING INFO
        // ========================================================

        const chess::SearchInfoCallback infoCallback =
            [](
                const SearchResult& info
                ) {
                    printUciSearchInfo(
                        info
                    );
            };

        stopAndJoinUciSearch();

        const Position searchPosition = position;
        const chess::SearchHistory searchHistory = history;

        chess::resetSearchStop();

        uciSearchThread = std::thread(
            [=]() mutable {
        SearchResult result;


        // ========================================================
        // EXPLICIT MOVETIME
        // ========================================================

        if (
            movetime > 0
            ) {
            result =
                chess::searchBestMoveTimed(
                    searchPosition,
                    movetime,
                    infoCallback,
                    searchHistory
                );
        }


        // ========================================================
        // NORMAL TOURNAMENT CLOCK
        // ========================================================

        else if (
            whiteTime >= 0 ||
            blackTime >= 0
            ) {
            const int allocatedTime =
                calculateMoveTime(
                    searchPosition,
                    whiteTime,
                    blackTime,
                    whiteIncrement,
                    blackIncrement,
                    movesToGo
                );


            if (
                allocatedTime > 0
                ) {
                result =
                    chess::searchBestMoveTimed(
                        searchPosition,
                        allocatedTime,
                        std::max(allocatedTime, allocatedTime * 2),
                        infoCallback,
                        searchHistory
                    );
            }

            else {
                result =
                    chess::searchBestMove(
                        searchPosition,
                        5,
                        infoCallback,
                        searchHistory
                    );
            }
        }


        // ========================================================
        // FIXED DEPTH
        // ========================================================

        else if (
            depth > 0
            ) {
            result =
                chess::searchBestMove(
                    searchPosition,
                    depth,
                    infoCallback,
                    searchHistory
                );
        }


        // ========================================================
        // TRUE INFINITE SEARCH (terminated by the UCI stop command)
        // ========================================================

        else if (
            infinite
            ) {
            result = chess::searchBestMove(
                searchPosition,
                127,
                infoCallback,
                searchHistory
            );
        }


        // ========================================================
        // FALLBACK
        // ========================================================

        else {
            result =
                chess::searchBestMove(
                    searchPosition,
                    5,
                    infoCallback,
                    searchHistory
                );
        }


        // ========================================================
        // FINAL UCI MOVE
        // ========================================================

        printUciBestMove(
            result
        );

            }
        );


        return true;
    }
} // anonymous namespace

void printEvaluationBreakdown(
    const chess::Position& pos
) {
    const chess::EvaluationBreakdown eval =
        chess::evaluateDetailed(
            pos
        );

    if (
        eval.insufficientMaterial
        ) {
        std::cout
            << "Insufficient material: draw\n"
            << "Final evaluation: 0.00\n";

        return;
    }

    auto printScore =
        [](
            const char* label,
            int mg,
            int eg
            ) {
                std::cout
                    << std::left
                    << std::setw(22)
                    << label
                    << "MG "
                    << std::showpos
                    << std::fixed
                    << std::setprecision(2)
                    << (mg / 100.0)
                    << "   EG "
                    << (eg / 100.0)
                    << std::noshowpos
                    << '\n';
        };

    std::cout
        << "\nEvaluation breakdown\n"
        << "------------------------------\n";

    printScore(
        "Material + PST:",
        eval.materialAndPstMg,
        eval.materialAndPstEg
    );

    printScore(
        "Bishop pair:",
        eval.bishopPairMg,
        eval.bishopPairEg
    );

    printScore(
        "Pawn structure:",
        eval.pawnStructureMg,
        eval.pawnStructureEg
    );

    printScore(
        "Piece activity:",
        eval.pieceActivityMg,
        eval.pieceActivityEg
    );

    printScore(
        "King safety:",
        eval.kingSafetyMg,
        eval.kingSafetyEg
    );

    printScore(
        "Tempo:",
        eval.tempoMg,
        eval.tempoEg
    );

    std::cout
        << "\nPhase: "
        << eval.phase
        << " / 24\n";

    const double mgPercent =
        (
            static_cast<double>(
                eval.phase
                )
            /
            24.0
            )
        *
        100.0;

    std::cout
        << "Blend: "
        << std::fixed
        << std::setprecision(1)
        << mgPercent
        << "% MG / "
        << (100.0 - mgPercent)
        << "% EG\n";

    std::cout
        << "Final evaluation: "
        << std::showpos
        << std::fixed
        << std::setprecision(2)
        << (eval.finalScore / 100.0)
        << std::noshowpos
        << '\n';
}
int main() {

    Position position =
        chess::startPosition();

    chess::SearchHistory gameHistory =
        {position.zobristKey};


    bool uciMode =
        false;

#ifdef _WIN32
    const bool interactiveConsole =
        _isatty(_fileno(stdin)) != 0;
#else
    const bool interactiveConsole =
        isatty(fileno(stdin)) != 0;
#endif


    // ========================================================
    // STARTUP BANNER
    // ========================================================

    if (interactiveConsole) {
        std::cout <<
            "=========================\n"
            "     KnightBot v" <<
            KNIGHTBOT_VERSION <<
            "\n"
            "=========================\n\n"
            "Incremental bitboard alpha-beta engine\n";

        printHelp();
        chess::printBoard(position);
    }


    std::string line;


    while (
        true
        ) {
        if (
            !uciMode &&
            interactiveConsole
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
            stopAndJoinUciSearch();
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

            const std::lock_guard<std::mutex> lock(uciOutputMutex);
            std::cout <<
                "id name KnightBot " <<
                KNIGHTBOT_VERSION <<
                '\n';


            std::cout <<
                "id author Joshua Wang\n";


            std::cout <<
                "option name UseNNUE type check default false\n"
                "option name EvalFile type string default <empty>\n";


            std::cout <<
                "uciok\n";


            continue;
        }


        if (
            command ==
            "isready"
            ) {
            const std::lock_guard<std::mutex> lock(uciOutputMutex);
            std::cout <<
                "readyok\n";

            continue;
        }


        if (
            command ==
            "ucinewgame"
            ) {
            stopAndJoinUciSearch();

            position =
                chess::startPosition();

            gameHistory = {position.zobristKey};


            chess::clearTranspositionTable();


            continue;
        }


        if (
            command ==
            "position"
            ) {
            stopAndJoinUciSearch();

            handleUciPosition(
                line,
                position,
                gameHistory
            );

            continue;
        }


        if (
            command ==
            "setoption"
            ) {
            stopAndJoinUciSearch();

            handleUciSetOption(
                line
            );

            continue;
        }


        if (
            command ==
            "go"
            ) {
            handleUciGo(
                line,
                position,
                gameHistory
            );

            continue;
        }

        if (command == "stop") {
            stopAndJoinUciSearch();
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

                    gameHistory.push_back(position.zobristKey);


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
                chess::evaluateActive(
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
// DETAILED STATIC EVAL
// ====================================================

        else if (
            command ==
            "evaldetail"
            ) {
                printEvaluationBreakdown(
                    position
                );
}


        else if (
            command ==
            "evalcompare"
            ) {
            const int handcrafted =
                chess::evaluate(
                    position
                );

            std::cout <<
                "\nEvaluation comparison\n"
                "------------------------------\n" <<
                std::showpos <<
                std::fixed <<
                std::setprecision(2) <<
                "Handcrafted: " <<
                (handcrafted / 100.0) <<
                '\n';

            if (
                chess::nnueLoaded()
                ) {
                const int nnue =
                    chess::evaluateNNUE(
                        position
                    );

                std::cout <<
                    "NNUE:        " <<
                    (nnue / 100.0) <<
                    '\n';
            }
            else {
                std::cout <<
                    std::noshowpos <<
                    "NNUE:        <not loaded>\n";
            }

            const int active =
                chess::evaluateActive(
                    position
                );

            std::cout <<
                std::showpos <<
                "Active:      " <<
                (active / 100.0) <<
                std::noshowpos <<
                '\n';
        }


            else if (
        command ==
        "nnueinfo"
        ) {
        std::cout <<
            "UseNNUE: " <<
            (chess::nnueEnabled() ? "true" : "false") <<
            '\n' <<
            "Loaded:  " <<
            (chess::nnueLoaded() ? "true" : "false") <<
            '\n';

        if (
            chess::nnueLoaded()
            ) {
            std::cout <<
                "File:    " <<
                chess::nnueLoadedFile() <<
                '\n' <<
                "Format:  " <<
                chess::nnueFormatName() <<
                '\n' <<
                "Version: " <<
                chess::nnueFormatVersion() <<
                '\n';

            if (
                chess::nnueFormatVersion() ==
                1
                ) {
                std::cout <<
                    "Inputs:  " <<
                    chess::NNUE_V1_INPUT_COUNT <<
                    '\n' <<
                    "Hidden:  " <<
                    chess::NNUE_V1_HIDDEN_COUNT <<
                    '\n';
            }
            else if (
                chess::nnueFormatVersion() ==
                2
                ) {
                std::cout <<
                    "Features:     " <<
                    chess::HALFKP_FEATURE_COUNT <<
                    '\n' <<
                    "Transformer:  " <<
                    chess::HALFKP_TRANSFORMER_HIDDEN <<
                    '\n' <<
                    "Context:      " <<
                    chess::HALFKP_CONTEXT_COUNT <<
                    '\n' <<
                    "Dense 1:      " <<
                    chess::HALFKP_DENSE1_COUNT <<
                    '\n' <<
                    "Dense 2:      " <<
                    chess::HALFKP_DENSE2_COUNT <<
                    '\n';
            }
        }
        }


        else if (
            command ==
            "nnueverify"
            ) {
            if (
                !chess::nnueLoaded()
                ) {
                std::cout <<
                    "No NNUE network loaded.\n";

                continue;
            }

            constexpr int TEST_GAMES =
                100;

            constexpr int MAX_MOVES =
                200;

            std::mt19937 rng(
                0x4B4E5545u
            );

            int positionsChecked =
                0;

            bool failed =
                false;


            for (
                int game = 0;
                game < TEST_GAMES &&
                !failed;
                ++game
                ) {
                chess::Position pos =
                    chess::startPosition();


                chess::rebuildHalfKPAccumulators(
                    pos
                );


                for (
                    int moveNumber = 0;
                    moveNumber < MAX_MOVES;
                    ++moveNumber
                    ) {
                    const int incremental =
                        chess::evaluateNNUE(
                            pos
                        );

                    const int rebuilt =
                        chess::evaluateHalfKPFullRebuild(
                            pos
                        );


                    ++positionsChecked;


                    if (
                        incremental !=
                        rebuilt
                        ) {
                        std::cout <<
                            "NNUE VERIFY FAILED\n" <<
                            "Game: " <<
                            game <<
                            '\n' <<
                            "Move: " <<
                            moveNumber <<
                            '\n' <<
                            "Incremental: " <<
                            incremental <<
                            '\n' <<
                            "Rebuilt:     " <<
                            rebuilt <<
                            '\n' <<
                            "FEN: " <<
                            chess::toFEN(
                                pos
                            ) <<
                            '\n';

                        failed =
                            true;

                        break;
                    }


                    chess::MoveList moves;

                    chess::generateLegalMoves(
                        pos,
                        moves
                    );


                    if (
                        moves.empty()
                        ) {
                        break;
                    }


                    std::uniform_int_distribution<
                        std::size_t
                    > distribution(
                        0,
                        moves.size() - 1
                    );


                    const chess::Move move =
                        moves[
                            distribution(
                                rng
                            )
                        ];


                    chess::UndoState undo;


                    chess::makeMove(
                        pos,
                        move,
                        undo
                    );


                    chess::updateHalfKPAfterMove(
                        pos,
                        move,
                        undo
                    );


                    const int afterMoveIncremental =
                        chess::evaluateNNUE(
                            pos
                        );

                    const int afterMoveRebuilt =
                        chess::evaluateHalfKPFullRebuild(
                            pos
                        );


                    ++positionsChecked;


                    if (
                        afterMoveIncremental !=
                        afterMoveRebuilt
                        ) {
                        std::cout <<
                            "NNUE VERIFY FAILED AFTER MOVE\n" <<
                            "Game: " <<
                            game <<
                            '\n' <<
                            "Move: " <<
                            moveNumber <<
                            '\n' <<
                            "Incremental: " <<
                            afterMoveIncremental <<
                            '\n' <<
                            "Rebuilt:     " <<
                            afterMoveRebuilt <<
                            '\n' <<
                            "FEN: " <<
                            chess::toFEN(
                                pos
                            ) <<
                            '\n';

                        failed =
                            true;

                        break;
                    }


                    chess::undoMove(
                        pos,
                        move,
                        undo
                    );


                    chess::updateHalfKPAfterUndo(
                        pos,
                        move,
                        undo
                    );


                    const int afterUndoIncremental =
                        chess::evaluateNNUE(
                            pos
                        );

                    const int afterUndoRebuilt =
                        chess::evaluateHalfKPFullRebuild(
                            pos
                        );


                    ++positionsChecked;


                    if (
                        afterUndoIncremental !=
                        afterUndoRebuilt
                        ) {
                        std::cout <<
                            "NNUE VERIFY FAILED AFTER UNDO\n" <<
                            "Game: " <<
                            game <<
                            '\n' <<
                            "Move: " <<
                            moveNumber <<
                            '\n' <<
                            "Incremental: " <<
                            afterUndoIncremental <<
                            '\n' <<
                            "Rebuilt:     " <<
                            afterUndoRebuilt <<
                            '\n' <<
                            "FEN: " <<
                            chess::toFEN(
                                pos
                            ) <<
                            '\n';

                        failed =
                            true;

                        break;
                    }


                    // Make it again so the random game can continue.
                    chess::makeMove(
                        pos,
                        move,
                        undo
                    );

                    chess::updateHalfKPAfterMove(
                        pos,
                        move,
                        undo
                    );
                }
            }


            if (
                !failed
                ) {
                std::cout <<
                    "NNUE VERIFY PASSED\n" <<
                    "Positions checked: " <<
                    positionsChecked <<
                    '\n';
            }
        }

        else if (
            command ==
            "nnue"
            ) {
            std::string setting;
            input >> setting;
            setting = lowerCopy(setting);

            if (setting == "on") {
                chess::setNNUEEnabled(true);
                chess::clearTranspositionTable();

                std::cout << "NNUE enabled";

                if (!chess::nnueLoaded()) {
                    std::cout <<
                        " (no network loaded; handcrafted fallback active)";
                }

                std::cout << ".\n";
            }
            else if (setting == "off") {
                chess::setNNUEEnabled(false);
                chess::clearTranspositionTable();
                std::cout << "NNUE disabled.\n";
            }
            else {
                std::cout << "Usage: nnue on|off\n";
            }
        }


        else if (
            command ==
            "nnueload"
            ) {
            std::string path;
            std::getline(input, path);
            path = trimCopy(path);

            if (path.empty()) {
                std::cout << "Usage: nnueload <network-file>\n";
                continue;
            }

            std::string error;

            if (chess::loadNNUE(path, &error)) {
                chess::clearTranspositionTable();
                std::cout << "Loaded NNUE network: " << path << '\n';
            }
            else {
                std::cout << "NNUE load failed: " << error << '\n';
            }
        }


        else if (
            command ==
            "nnuegentest"
            ) {
            std::string path;
            std::getline(input, path);
            path = trimCopy(path);

            if (path.empty()) {
                path = "knightbot-test.nnue";
            }

            std::string error;

            if (chess::writeNNUECompatibilityTestNetwork(path, &error)) {
                std::cout << "Created compatibility network: " << path << '\n';
            }
            else {
                std::cout << "Unable to create test network: " << error << '\n';
            }
        }


        else if (
            command ==
            "exportpos"
            ) {
            std::string path;
            input >> path;

            if (path.empty()) {
                std::cout << "Usage: exportpos <file> [target_cp]\n";
                continue;
            }

            int target = chess::evaluate(position);
            int suppliedTarget = 0;

            if (input >> suppliedTarget) {
                target = suppliedTarget;
            }

            std::string error;

            if (chess::appendTrainingPosition(
                    path,
                    position,
                    target,
                    &error
                )) {
                std::cout <<
                    "Exported training position to " << path <<
                    " with target " << target << " cp.\n";
            }
            else {
                std::cout << "Training export failed: " << error << '\n';
            }
        }


        else if (
            command ==
            "selftest"
            ) {
            stopAndJoinUciSearch();
            chess::runSelfTest();
        }


        else if (
            command ==
            "bench"
            ) {
            stopAndJoinUciSearch();
            chess::runBenchmark();
        }


        else if (
            command ==
            "benchfull"
            ) {
            stopAndJoinUciSearch();
            chess::runFullBenchmark();
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

                gameHistory.push_back(position.zobristKey);


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

                gameHistory.push_back(position.zobristKey);


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

            gameHistory = {position.zobristKey};


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

                gameHistory = {position.zobristKey};


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
        !uciMode &&
        interactiveConsole
        ) {
        std::cout <<
            "\nKnightBot stopped.\n";
    }

    stopAndJoinUciSearch();


    return 0;
}
