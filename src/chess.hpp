#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace chess {

    using Bitboard = std::uint64_t;

    constexpr int MAX_MOVES = 256;

    enum PieceIndex {
        WP = 0,
        WN,
        WB,
        WR,
        WQ,
        WK,

        BP,
        BN,
        BB,
        BR,
        BQ,
        BK,

        PIECE_COUNT
    };

    enum OccupancyIndex {
        WHITE_OCC = 0,
        BLACK_OCC = 1,
        BOTH_OCC = 2
    };

    struct Move {
        int from = -1;
        int to = -1;

        char promotion = 0;

        bool enPassant = false;
        bool castle = false;
    };

    struct MoveList {
        std::array<Move, MAX_MOVES> moves{};
        int count = 0;

        void clear() {
            count = 0;
        }

        void push_back(
            const Move& move
        ) {
            if (count < MAX_MOVES) {
                moves[count++] = move;
            }
        }

        std::size_t size() const {
            return static_cast<std::size_t>(
                count
                );
        }

        bool empty() const {
            return count == 0;
        }

        Move& operator[](
            std::size_t index
            ) {
            return moves[index];
        }

        const Move& operator[](
            std::size_t index
            ) const {
            return moves[index];
        }

        Move& front() {
            return moves[0];
        }

        const Move& front() const {
            return moves[0];
        }

        Move* begin() {
            return moves.data();
        }

        Move* end() {
            return moves.data() + count;
        }

        const Move* begin() const {
            return moves.data();
        }

        const Move* end() const {
            return moves.data() + count;
        }
    };

    struct UndoState {
        char movedPiece = '.';

        char capturedPiece = '.';
        int capturedSquare = -1;

        bool castleWK = false;
        bool castleWQ = false;
        bool castleBK = false;
        bool castleBQ = false;

        int enPassantSquare = -1;

        int halfmoveClock = 0;
        int fullmoveNumber = 1;

        std::uint64_t zobristKey = 0;
    };

    struct Position {
        std::array<char, 64> board{};

        std::array<
            Bitboard,
            PIECE_COUNT
        > pieces{};

        std::array<
            Bitboard,
            3
        > occupancy{};

        bool whiteToMove = true;

        bool castleWK = false;
        bool castleWQ = false;
        bool castleBK = false;
        bool castleBQ = false;

        int enPassantSquare = -1;

        int halfmoveClock = 0;
        int fullmoveNumber = 1;

        std::uint64_t zobristKey = 0;
                // ========================================================
        // HALFKP-512 INCREMENTAL CACHE
        // ========================================================
        //
        // These are transient evaluation caches. They are not part
        // of the chess position itself and are not included in FEN
        // or Zobrist hashing.
        //
        // A generation number prevents using accumulators created
        // with an older/different NNUE network.
        // ========================================================

        std::array<
            std::int32_t,
            512
        > halfKPWhiteAccumulator{};

        std::array<
            std::int32_t,
            512
        > halfKPBlackAccumulator{};

        int halfKPWhiteKingSquare = -1;
        int halfKPBlackKingSquare = -1;

        std::uint64_t halfKPGeneration = 0;

        bool halfKPValid = false;
    };

    Position startPosition();

    Position fromFEN(
        const std::string& fen
    );

    std::string toFEN(
        const Position& pos
    );

    void printBoard(
        const Position& pos
    );

    void rebuildBitboards(
        Position& pos
    );

    std::uint64_t calculateZobrist(
        const Position& pos
    );

    MoveList generateLegalMoves(
        const Position& pos
    );

    void generateLegalMoves(
        Position& pos,
        MoveList& legal
    );

    void makeMove(
        Position& pos,
        const Move& move
    );

    void makeMove(
        Position& pos,
        const Move& move,
        UndoState& undo
    );

    void undoMove(
        Position& pos,
        const Move& move,
        const UndoState& undo
    );

    bool isSquareAttacked(
        const Position& pos,
        int square,
        bool byWhite
    );

    bool inCheck(
        const Position& pos,
        bool whiteKing
    );

    // Draw by insufficient mating material.
    bool isInsufficientMaterial(
        const Position& pos
    );

    std::uint64_t perft(
        const Position& pos,
        int depth
    );

    std::string squareName(
        int square
    );

    std::string moveToUci(
        const Move& move
    );

    std::string moveToSan(
        const Position& pos,
        const Move& move
    );

} // namespace chess