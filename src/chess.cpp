#include "chess.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace chess {

    namespace {

        constexpr const char* START_FEN =
            "rnbqkbnr/pppppppp/8/8/8/8/"
            "PPPPPPPP/RNBQKBNR w KQkq - 0 1";


        // ============================================================
        // BASIC HELPERS
        // ============================================================

        int sq(int file, int rank) {
            return rank * 8 + file;
        }

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

        Bitboard bit(int square) {
            return
                1ULL << square;
        }

        int popLsb(Bitboard& board) {

            const int square =
                static_cast<int>(
                    std::countr_zero(board)
                    );

            board &=
                board - 1;

            return square;
        }

        bool isWhite(char piece) {

            return
                piece >= 'A' &&
                piece <= 'Z';
        }

        bool isBlack(char piece) {

            return
                piece >= 'a' &&
                piece <= 'z';
        }

        bool enemyPiece(
            char piece,
            bool white
        ) {
            return
                white
                ? isBlack(piece)
                : isWhite(piece);
        }

        int pieceIndex(char piece) {

            switch (piece) {

            case 'P': return WP;
            case 'N': return WN;
            case 'B': return WB;
            case 'R': return WR;
            case 'Q': return WQ;
            case 'K': return WK;

            case 'p': return BP;
            case 'n': return BN;
            case 'b': return BB;
            case 'r': return BR;
            case 'q': return BQ;
            case 'k': return BK;

            default:
                return -1;
            }
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


        // ============================================================
        // ZOBRIST
        // ============================================================

        std::uint64_t splitmix64(
            std::uint64_t& state
        ) {
            std::uint64_t z =
                (
                    state +=
                    0x9E3779B97F4A7C15ULL
                    );

            z =
                (
                    z ^
                    (z >> 30)
                    )
                *
                0xBF58476D1CE4E5B9ULL;

            z =
                (
                    z ^
                    (z >> 27)
                    )
                *
                0x94D049BB133111EBULL;

            return
                z ^
                (z >> 31);
        }

        struct ZobristTables {

            std::uint64_t
                pieces[PIECE_COUNT][64]{};

            std::uint64_t side{};

            std::uint64_t castling[16]{};

            std::uint64_t enPassantFile[8]{};

            ZobristTables() {

                std::uint64_t seed =
                    0x4B4E49474854424FULL;

                for (
                    int piece = 0;
                    piece < PIECE_COUNT;
                    ++piece
                    ) {
                    for (
                        int square = 0;
                        square < 64;
                        ++square
                        ) {
                        pieces[piece][square] =
                            splitmix64(seed);
                    }
                }

                side =
                    splitmix64(seed);

                for (
                    auto& value :
                    castling
                    ) {
                    value =
                        splitmix64(seed);
                }

                for (
                    auto& value :
                    enPassantFile
                    ) {
                    value =
                        splitmix64(seed);
                }
            }
        };

        const ZobristTables ZOBRIST;


        int castlingIndex(
            const Position& pos
        ) {
            int result = 0;

            if (pos.castleWK) {
                result |= 1;
            }

            if (pos.castleWQ) {
                result |= 2;
            }

            if (pos.castleBK) {
                result |= 4;
            }

            if (pos.castleBQ) {
                result |= 8;
            }

            return result;
        }


        // ============================================================
        // ATTACK TABLES
        // ============================================================

        std::array<Bitboard, 64>
            createKnightAttacks() {

            std::array<Bitboard, 64>
                table{};

            constexpr int jumps[8][2] = {
                { 1,  2},
                { 2,  1},
                { 2, -1},
                { 1, -2},
                {-1, -2},
                {-2, -1},
                {-2,  1},
                {-1,  2}
            };

            for (
                int square = 0;
                square < 64;
                ++square
                ) {
                const int file =
                    fileOf(square);

                const int rank =
                    rankOf(square);

                for (
                    const auto& jump :
                    jumps
                    ) {
                    const int f =
                        file + jump[0];

                    const int r =
                        rank + jump[1];

                    if (
                        onBoard(f, r)
                        ) {
                        table[square] |=
                            bit(
                                sq(f, r)
                            );
                    }
                }
            }

            return table;
        }


        std::array<Bitboard, 64>
            createKingAttacks() {

            std::array<Bitboard, 64>
                table{};

            for (
                int square = 0;
                square < 64;
                ++square
                ) {
                const int file =
                    fileOf(square);

                const int rank =
                    rankOf(square);

                for (
                    int df = -1;
                    df <= 1;
                    ++df
                    ) {
                    for (
                        int dr = -1;
                        dr <= 1;
                        ++dr
                        ) {
                        if (
                            df == 0 &&
                            dr == 0
                            ) {
                            continue;
                        }

                        const int f =
                            file + df;

                        const int r =
                            rank + dr;

                        if (
                            onBoard(f, r)
                            ) {
                            table[square] |=
                                bit(
                                    sq(f, r)
                                );
                        }
                    }
                }
            }

            return table;
        }


        std::array<
            std::array<Bitboard, 64>,
            2
        >
            createPawnAttacks() {

            std::array<
                std::array<Bitboard, 64>,
                2
            > table{};

            for (
                int square = 0;
                square < 64;
                ++square
                ) {
                const int file =
                    fileOf(square);

                const int rank =
                    rankOf(square);

                for (
                    int df :
                {-1, 1}
                    ) {
                    const int wf =
                        file + df;

                    const int wr =
                        rank + 1;

                    if (
                        onBoard(wf, wr)
                        ) {
                        table[0][square] |=
                            bit(
                                sq(wf, wr)
                            );
                    }

                    const int bf =
                        file + df;

                    const int br =
                        rank - 1;

                    if (
                        onBoard(bf, br)
                        ) {
                        table[1][square] |=
                            bit(
                                sq(bf, br)
                            );
                    }
                }
            }

            return table;
        }


        const auto KNIGHT_ATTACKS =
            createKnightAttacks();

        const auto KING_ATTACKS =
            createKingAttacks();

        const auto PAWN_ATTACKS =
            createPawnAttacks();


        // ============================================================
        // INCREMENTAL PIECE / OCCUPANCY UPDATES
        // ============================================================

        void removePiece(
            Position& pos,
            int square
        ) {
            const char piece =
                pos.board[square];

            if (
                piece == '.'
                ) {
                return;
            }

            const int index =
                pieceIndex(piece);

            const Bitboard mask =
                bit(square);

            pos.board[square] =
                '.';

            pos.pieces[index] &=
                ~mask;

            if (
                isWhite(piece)
                ) {
                pos.occupancy[WHITE_OCC] &=
                    ~mask;
            }

            else {
                pos.occupancy[BLACK_OCC] &=
                    ~mask;
            }

            pos.occupancy[BOTH_OCC] &=
                ~mask;

            // Incremental hash.
            pos.zobristKey ^=
                ZOBRIST.pieces[
                    index
                ][
                    square
                ];
        }


        void addPiece(
            Position& pos,
            int square,
            char piece
        ) {
            const int index =
                pieceIndex(piece);

            const Bitboard mask =
                bit(square);

            pos.board[square] =
                piece;

            pos.pieces[index] |=
                mask;

            if (
                isWhite(piece)
                ) {
                pos.occupancy[WHITE_OCC] |=
                    mask;
            }

            else {
                pos.occupancy[BLACK_OCC] |=
                    mask;
            }

            pos.occupancy[BOTH_OCC] |=
                mask;

            pos.zobristKey ^=
                ZOBRIST.pieces[
                    index
                ][
                    square
                ];
        }


        // ============================================================
        // PROMOTIONS
        // ============================================================

        void addPromotionMoves(
            MoveList& moves,
            int from,
            int to
        ) {
            for (
                char promotion :
            {'q', 'r', 'b', 'n'}
                ) {
                Move move;

                move.from = from;
                move.to = to;
                move.promotion = promotion;

                moves.push_back(move);
            }
        }


        // ============================================================
        // SLIDERS
        // ============================================================

        void addSlidingMoves(
            const Position& pos,
            MoveList& moves,
            int from,
            bool white,
            const int directions[][2],
            int directionCount
        ) {
            const int file =
                fileOf(from);

            const int rank =
                rankOf(from);

            for (
                int direction = 0;
                direction < directionCount;
                ++direction
                ) {
                int f =
                    file +
                    directions[direction][0];

                int r =
                    rank +
                    directions[direction][1];

                while (
                    onBoard(f, r)
                    ) {
                    const int to =
                        sq(f, r);

                    const char target =
                        pos.board[to];

                    if (
                        target == '.'
                        ) {
                        moves.push_back(
                            Move{
                                from,
                                to
                            }
                        );
                    }

                    else {
                        if (
                            enemyPiece(
                                target,
                                white
                            )
                            ) {
                            moves.push_back(
                                Move{
                                    from,
                                    to
                                }
                            );
                        }

                        break;
                    }

                    f +=
                        directions[direction][0];

                    r +=
                        directions[direction][1];
                }
            }
        }


        // ============================================================
        // PSEUDO-LEGAL GENERATION
        // ============================================================

        void generatePseudoLegalMoves(
            const Position& pos,
            MoveList& moves
        ) {
            moves.clear();

            const bool white =
                pos.whiteToMove;

            const Bitboard own =
                pos.occupancy[
                    white
                        ? WHITE_OCC
                        : BLACK_OCC
                ];


            // ========================================================
            // PAWNS
            // ========================================================

            Bitboard pawns =
                pos.pieces[
                    white
                        ? WP
                        : BP
                ];

            while (pawns) {

                const int from =
                    popLsb(pawns);

                const int file =
                    fileOf(from);

                const int rank =
                    rankOf(from);

                const int direction =
                    white
                    ? 1
                    : -1;

                const int startRank =
                    white
                    ? 1
                    : 6;

                const int promotionRank =
                    white
                    ? 7
                    : 0;


                const int oneRank =
                    rank + direction;

                if (
                    onBoard(
                        file,
                        oneRank
                    )
                    ) {
                    const int one =
                        sq(
                            file,
                            oneRank
                        );

                    if (
                        pos.board[one] == '.'
                        ) {
                        if (
                            oneRank ==
                            promotionRank
                            ) {
                            addPromotionMoves(
                                moves,
                                from,
                                one
                            );
                        }

                        else {
                            moves.push_back(
                                Move{
                                    from,
                                    one
                                }
                            );

                            if (
                                rank ==
                                startRank
                                ) {
                                const int two =
                                    sq(
                                        file,
                                        rank +
                                        2 *
                                        direction
                                    );

                                if (
                                    pos.board[two] ==
                                    '.'
                                    ) {
                                    moves.push_back(
                                        Move{
                                            from,
                                            two
                                        }
                                    );
                                }
                            }
                        }
                    }
                }


                Bitboard attacks =
                    PAWN_ATTACKS[
                        white
                            ? 0
                            : 1
                    ][
                        from
                    ];

                while (attacks) {

                    const int to =
                        popLsb(attacks);

                    if (
                        enemyPiece(
                            pos.board[to],
                            white
                        )
                        ||
                        to ==
                        pos.enPassantSquare
                        ) {
                        if (
                            rankOf(to) ==
                            promotionRank
                            ) {
                            addPromotionMoves(
                                moves,
                                from,
                                to
                            );
                        }

                        else {
                            Move move{
                                from,
                                to
                            };

                            if (
                                to ==
                                pos.enPassantSquare &&
                                pos.board[to] ==
                                '.'
                                ) {
                                move.enPassant =
                                    true;
                            }

                            moves.push_back(
                                move
                            );
                        }
                    }
                }
            }


            // ========================================================
            // KNIGHTS
            // ========================================================

            Bitboard knights =
                pos.pieces[
                    white
                        ? WN
                        : BN
                ];

            while (knights) {

                const int from =
                    popLsb(knights);

                Bitboard attacks =
                    KNIGHT_ATTACKS[from]
                    &
                    ~own;

                while (attacks) {

                    const int to =
                        popLsb(attacks);

                    moves.push_back(
                        Move{
                            from,
                            to
                        }
                    );
                }
            }


            // ========================================================
            // BISHOPS
            // ========================================================

            constexpr int bishopDirs[4][2] = {
                { 1, 1},
                { 1,-1},
                {-1, 1},
                {-1,-1}
            };

            Bitboard bishops =
                pos.pieces[
                    white
                        ? WB
                        : BB
                ];

            while (bishops) {

                const int from =
                    popLsb(bishops);

                addSlidingMoves(
                    pos,
                    moves,
                    from,
                    white,
                    bishopDirs,
                    4
                );
            }


            // ========================================================
            // ROOKS
            // ========================================================

            constexpr int rookDirs[4][2] = {
                { 1, 0},
                {-1, 0},
                { 0, 1},
                { 0,-1}
            };

            Bitboard rooks =
                pos.pieces[
                    white
                        ? WR
                        : BR
                ];

            while (rooks) {

                const int from =
                    popLsb(rooks);

                addSlidingMoves(
                    pos,
                    moves,
                    from,
                    white,
                    rookDirs,
                    4
                );
            }


            // ========================================================
            // QUEENS
            // ========================================================

            constexpr int queenDirs[8][2] = {
                { 1, 1},
                { 1,-1},
                {-1, 1},
                {-1,-1},

                { 1, 0},
                {-1, 0},
                { 0, 1},
                { 0,-1}
            };

            Bitboard queens =
                pos.pieces[
                    white
                        ? WQ
                        : BQ
                ];

            while (queens) {

                const int from =
                    popLsb(queens);

                addSlidingMoves(
                    pos,
                    moves,
                    from,
                    white,
                    queenDirs,
                    8
                );
            }


            // ========================================================
            // KING
            // ========================================================

            Bitboard king =
                pos.pieces[
                    white
                        ? WK
                        : BK
                ];

            if (king) {

                const int from =
                    static_cast<int>(
                        std::countr_zero(
                            king
                        )
                        );

                Bitboard attacks =
                    KING_ATTACKS[from]
                    &
                    ~own;

                while (attacks) {

                    const int to =
                        popLsb(attacks);

                    moves.push_back(
                        Move{
                            from,
                            to
                        }
                    );
                }


                // WHITE CASTLING
                if (
                    white &&
                    from == 4
                    ) {
                    if (
                        pos.castleWK &&
                        pos.board[5] == '.' &&
                        pos.board[6] == '.' &&
                        pos.board[7] == 'R' &&
                        !isSquareAttacked(
                            pos,
                            4,
                            false
                        ) &&
                        !isSquareAttacked(
                            pos,
                            5,
                            false
                        ) &&
                        !isSquareAttacked(
                            pos,
                            6,
                            false
                        )
                        ) {
                        Move move{
                            4,
                            6
                        };

                        move.castle = true;

                        moves.push_back(move);
                    }


                    if (
                        pos.castleWQ &&
                        pos.board[1] == '.' &&
                        pos.board[2] == '.' &&
                        pos.board[3] == '.' &&
                        pos.board[0] == 'R' &&
                        !isSquareAttacked(
                            pos,
                            4,
                            false
                        ) &&
                        !isSquareAttacked(
                            pos,
                            3,
                            false
                        ) &&
                        !isSquareAttacked(
                            pos,
                            2,
                            false
                        )
                        ) {
                        Move move{
                            4,
                            2
                        };

                        move.castle = true;

                        moves.push_back(move);
                    }
                }


                // BLACK CASTLING
                if (
                    !white &&
                    from == 60
                    ) {
                    if (
                        pos.castleBK &&
                        pos.board[61] == '.' &&
                        pos.board[62] == '.' &&
                        pos.board[63] == 'r' &&
                        !isSquareAttacked(
                            pos,
                            60,
                            true
                        ) &&
                        !isSquareAttacked(
                            pos,
                            61,
                            true
                        ) &&
                        !isSquareAttacked(
                            pos,
                            62,
                            true
                        )
                        ) {
                        Move move{
                            60,
                            62
                        };

                        move.castle = true;

                        moves.push_back(move);
                    }


                    if (
                        pos.castleBQ &&
                        pos.board[57] == '.' &&
                        pos.board[58] == '.' &&
                        pos.board[59] == '.' &&
                        pos.board[56] == 'r' &&
                        !isSquareAttacked(
                            pos,
                            60,
                            true
                        ) &&
                        !isSquareAttacked(
                            pos,
                            59,
                            true
                        ) &&
                        !isSquareAttacked(
                            pos,
                            58,
                            true
                        )
                        ) {
                        Move move{
                            60,
                            58
                        };

                        move.castle = true;

                        moves.push_back(move);
                    }
                }
            }
        }

    } // anonymous namespace


    // ============================================================
    // ZOBRIST RECOMPUTATION
    // ============================================================

    std::uint64_t calculateZobrist(
        const Position& pos
    ) {
        std::uint64_t hash = 0;

        for (
            int piece = 0;
            piece < PIECE_COUNT;
            ++piece
            ) {
            Bitboard pieces =
                pos.pieces[piece];

            while (pieces) {

                const int square =
                    popLsb(pieces);

                hash ^=
                    ZOBRIST.pieces[
                        piece
                    ][
                        square
                    ];
            }
        }

        if (
            !pos.whiteToMove
            ) {
            hash ^=
                ZOBRIST.side;
        }

        hash ^=
            ZOBRIST.castling[
                castlingIndex(pos)
            ];

        if (
            pos.enPassantSquare >= 0
            ) {
            hash ^=
                ZOBRIST.enPassantFile[
                    fileOf(
                        pos.enPassantSquare
                    )
                ];
        }

        return hash;
    }


    // ============================================================
    // REBUILD
    // ============================================================

    void rebuildBitboards(
        Position& pos
    ) {
        pos.pieces.fill(0);
        pos.occupancy.fill(0);

        for (
            int square = 0;
            square < 64;
            ++square
            ) {
            const char piece =
                pos.board[square];

            const int index =
                pieceIndex(piece);

            if (
                index < 0
                ) {
                continue;
            }

            const Bitboard mask =
                bit(square);

            pos.pieces[index] |=
                mask;

            if (
                isWhite(piece)
                ) {
                pos.occupancy[
                    WHITE_OCC
                ] |=
                    mask;
            }

            else {
                pos.occupancy[
                    BLACK_OCC
                ] |=
                    mask;
            }
        }

        pos.occupancy[
            BOTH_OCC
        ] =
            pos.occupancy[
                WHITE_OCC
            ]
                |
                pos.occupancy[
                    BLACK_OCC
                ];

            pos.zobristKey =
                calculateZobrist(pos);
    }


    // ============================================================
    // FEN
    // ============================================================

    Position startPosition() {

        return
            fromFEN(
                START_FEN
            );
    }


    Position fromFEN(
        const std::string& fen
    ) {
        Position pos;

        pos.board.fill('.');

        std::istringstream input(
            fen
        );

        std::string boardPart;
        std::string side;
        std::string castling;
        std::string enPassant;

        if (!(
            input >>
            boardPart >>
            side >>
            castling >>
            enPassant >>
            pos.halfmoveClock >>
            pos.fullmoveNumber
            )) {
            throw std::runtime_error(
                "Invalid FEN"
            );
        }

        int rank = 7;
        int file = 0;

        for (
            char c :
        boardPart
            ) {
            if (
                c == '/'
                ) {
                if (
                    file != 8
                    ) {
                    throw std::runtime_error(
                        "Invalid FEN row"
                    );
                }

                --rank;
                file = 0;

                continue;
            }

            if (
                std::isdigit(
                    static_cast<unsigned char>(c)
                )
                ) {
                file +=
                    c - '0';

                if (
                    file > 8
                    ) {
                    throw std::runtime_error(
                        "Invalid FEN row"
                    );
                }

                continue;
            }

            if (
                pieceIndex(c) < 0 ||
                !onBoard(
                    file,
                    rank
                )
                ) {
                throw std::runtime_error(
                    "Invalid FEN board"
                );
            }

            pos.board[
                sq(file, rank)
            ] =
                c;

                ++file;
        }

        if (
            rank != 0 ||
            file != 8
            ) {
            throw std::runtime_error(
                "Invalid FEN board dimensions"
            );
        }

        if (
            side == "w"
            ) {
            pos.whiteToMove = true;
        }

        else if (
            side == "b"
            ) {
            pos.whiteToMove = false;
        }

        else {
            throw std::runtime_error(
                "Invalid side"
            );
        }


        pos.castleWK =
            castling.find('K') !=
            std::string::npos;

        pos.castleWQ =
            castling.find('Q') !=
            std::string::npos;

        pos.castleBK =
            castling.find('k') !=
            std::string::npos;

        pos.castleBQ =
            castling.find('q') !=
            std::string::npos;


        if (
            enPassant == "-"
            ) {
            pos.enPassantSquare =
                -1;
        }

        else {
            if (
                enPassant.size() != 2 ||
                enPassant[0] < 'a' ||
                enPassant[0] > 'h' ||
                enPassant[1] < '1' ||
                enPassant[1] > '8'
                ) {
                throw std::runtime_error(
                    "Invalid en passant"
                );
            }

            pos.enPassantSquare =
                sq(
                    enPassant[0] - 'a',
                    enPassant[1] - '1'
                );
        }

        rebuildBitboards(pos);

        return pos;
    }


    std::string toFEN(
        const Position& pos
    ) {
        std::ostringstream output;

        for (
            int rank = 7;
            rank >= 0;
            --rank
            ) {
            int empty = 0;

            for (
                int file = 0;
                file < 8;
                ++file
                ) {
                const char piece =
                    pos.board[
                        sq(file, rank)
                    ];

                if (
                    piece == '.'
                    ) {
                    ++empty;
                }

                else {
                    if (
                        empty > 0
                        ) {
                        output <<
                            empty;

                        empty = 0;
                    }

                    output <<
                        piece;
                }
            }

            if (
                empty > 0
                ) {
                output <<
                    empty;
            }

            if (
                rank != 0
                ) {
                output <<
                    '/';
            }
        }

        output <<
            (
                pos.whiteToMove
                ? " w "
                : " b "
                );

        std::string castling;

        if (pos.castleWK) castling += 'K';
        if (pos.castleWQ) castling += 'Q';
        if (pos.castleBK) castling += 'k';
        if (pos.castleBQ) castling += 'q';

        output <<
            (
                castling.empty()
                ? "-"
                : castling
                )
            <<
            ' ';

        if (
            pos.enPassantSquare < 0
            ) {
            output <<
                '-';
        }

        else {
            output <<
                squareName(
                    pos.enPassantSquare
                );
        }

        output <<
            ' ' <<
            pos.halfmoveClock <<
            ' ' <<
            pos.fullmoveNumber;

        return output.str();
    }


    // ============================================================
    // BOARD
    // ============================================================

    void printBoard(
        const Position& pos
    ) {
        std::cout <<
            '\n';

        for (
            int rank = 7;
            rank >= 0;
            --rank
            ) {
            std::cout <<
                rank + 1 <<
                "  ";

            for (
                int file = 0;
                file < 8;
                ++file
                ) {
                std::cout <<
                    pos.board[
                        sq(file, rank)
                    ]
                    <<
                    ' ';
            }

            std::cout <<
                '\n';
        }

        std::cout <<
            "\n   a b c d e f g h\n\n";

        std::cout <<
            "Side to move: " <<
            (
                pos.whiteToMove
                ? "White"
                : "Black"
                )
            <<
            '\n';
    }


    // ============================================================
    // ATTACK DETECTION
    // ============================================================

    bool isSquareAttacked(
        const Position& pos,
        int square,
        bool byWhite
    ) {
        const Bitboard target =
            bit(square);


        // Pawns.
        //
        // Reverse lookup:
        // if a target is attacked by White,
        // see if any white pawn exists on a square
        // from which a white pawn attacks target.
        Bitboard pawns =
            pos.pieces[
                byWhite
                    ? WP
                    : BP
            ];

        while (pawns) {

            const int pawnSquare =
                popLsb(pawns);

            if (
                PAWN_ATTACKS[
                    byWhite
                        ? 0
                        : 1
                ][
                    pawnSquare
                ]
                        &
                        target
                        ) {
                return true;
            }
        }


        // Knights.
        if (
            KNIGHT_ATTACKS[square]
            &
            pos.pieces[
                byWhite
                    ? WN
                    : BN
            ]
            ) {
            return true;
        }


        // King.
        if (
            KING_ATTACKS[square]
            &
            pos.pieces[
                byWhite
                    ? WK
                    : BK
            ]
            ) {
            return true;
        }


        const int file =
            fileOf(square);

        const int rank =
            rankOf(square);


        // Bishop / Queen.
        constexpr int bishopDirs[4][2] = {
            { 1, 1},
            { 1,-1},
            {-1, 1},
            {-1,-1}
        };

        for (
            const auto& direction :
            bishopDirs
            ) {
            int f =
                file +
                direction[0];

            int r =
                rank +
                direction[1];

            while (
                onBoard(f, r)
                ) {
                const char piece =
                    pos.board[
                        sq(f, r)
                    ];

                if (
                    piece != '.'
                    ) {
                    if (
                        byWhite &&
                        (
                            piece == 'B' ||
                            piece == 'Q'
                            )
                        ) {
                        return true;
                    }

                    if (
                        !byWhite &&
                        (
                            piece == 'b' ||
                            piece == 'q'
                            )
                        ) {
                        return true;
                    }

                    break;
                }

                f += direction[0];
                r += direction[1];
            }
        }


        // Rook / Queen.
        constexpr int rookDirs[4][2] = {
            { 1,0},
            {-1,0},
            { 0,1},
            { 0,-1}
        };

        for (
            const auto& direction :
            rookDirs
            ) {
            int f =
                file +
                direction[0];

            int r =
                rank +
                direction[1];

            while (
                onBoard(f, r)
                ) {
                const char piece =
                    pos.board[
                        sq(f, r)
                    ];

                if (
                    piece != '.'
                    ) {
                    if (
                        byWhite &&
                        (
                            piece == 'R' ||
                            piece == 'Q'
                            )
                        ) {
                        return true;
                    }

                    if (
                        !byWhite &&
                        (
                            piece == 'r' ||
                            piece == 'q'
                            )
                        ) {
                        return true;
                    }

                    break;
                }

                f += direction[0];
                r += direction[1];
            }
        }

        return false;
    }


    bool inCheck(
        const Position& pos,
        bool whiteKing
    ) {
        const Bitboard king =
            pos.pieces[
                whiteKing
                    ? WK
                    : BK
            ];

        if (!king) {
            return true;
        }

        const int kingSquare =
            static_cast<int>(
                std::countr_zero(
                    king
                )
                );

        return
            isSquareAttacked(
                pos,
                kingSquare,
                !whiteKing
            );
    }
    // ============================================================
// INSUFFICIENT MATERIAL
// ============================================================

    bool isInsufficientMaterial(
        const Position& pos
    ) {
        int whiteKnights = 0;
        int blackKnights = 0;

        int whiteBishops = 0;
        int blackBishops = 0;

        int whiteBishopSquare = -1;
        int blackBishopSquare = -1;


        for (
            int square = 0;
            square < 64;
            ++square
            ) {
            const char piece =
                pos.board[square];


            switch (piece) {

                // Any pawn, rook, or queen means this is not one
                // of our insufficient-material cases.
            case 'P':
            case 'p':
            case 'R':
            case 'r':
            case 'Q':
            case 'q':
                return false;


            case 'N':
                ++whiteKnights;
                break;


            case 'n':
                ++blackKnights;
                break;


            case 'B':
                ++whiteBishops;
                whiteBishopSquare =
                    square;
                break;


            case 'b':
                ++blackBishops;
                blackBishopSquare =
                    square;
                break;


            default:
                break;
            }
        }


        const int totalKnights =
            whiteKnights +
            blackKnights;


        const int totalBishops =
            whiteBishops +
            blackBishops;


        const int totalMinorPieces =
            totalKnights +
            totalBishops;


        // ========================================================
        // K vs K
        // ========================================================

        if (
            totalMinorPieces == 0
            ) {
            return true;
        }


        // ========================================================
        // K+B vs K
        // K+N vs K
        // ========================================================

        if (
            totalMinorPieces == 1
            ) {
            return true;
        }


        // ========================================================
        // K+B vs K+B
        //
        // If each side has exactly one bishop and both bishops
        // live on the same colour complex, mate is impossible.
        // ========================================================

        if (
            totalKnights == 0 &&
            whiteBishops == 1 &&
            blackBishops == 1
            ) {
            const int whiteFile =
                whiteBishopSquare & 7;

            const int whiteRank =
                whiteBishopSquare >> 3;


            const int blackFile =
                blackBishopSquare & 7;

            const int blackRank =
                blackBishopSquare >> 3;


            const bool whiteBishopDark =
                (
                    (
                        whiteFile +
                        whiteRank
                        ) &
                    1
                    ) != 0;


            const bool blackBishopDark =
                (
                    (
                        blackFile +
                        blackRank
                        ) &
                    1
                    ) != 0;


            if (
                whiteBishopDark ==
                blackBishopDark
                ) {
                return true;
            }
        }


        return false;
    }

    // ============================================================
    // MAKE MOVE
    // ============================================================

    void makeMove(
        Position& pos,
        const Move& move,
        UndoState& undo
    ) {
        undo.movedPiece =
            pos.board[
                move.from
            ];

        undo.castleWK =
            pos.castleWK;

        undo.castleWQ =
            pos.castleWQ;

        undo.castleBK =
            pos.castleBK;

        undo.castleBQ =
            pos.castleBQ;

        undo.enPassantSquare =
            pos.enPassantSquare;

        undo.halfmoveClock =
            pos.halfmoveClock;

        undo.fullmoveNumber =
            pos.fullmoveNumber;

        undo.zobristKey =
            pos.zobristKey;


        const char movingPiece =
            undo.movedPiece;

        const bool white =
            isWhite(
                movingPiece
            );


        // Remove old non-piece hash state.
        pos.zobristKey ^=
            ZOBRIST.castling[
                castlingIndex(pos)
            ];

        if (
            pos.enPassantSquare >= 0
            ) {
            pos.zobristKey ^=
                ZOBRIST.enPassantFile[
                    fileOf(
                        pos.enPassantSquare
                    )
                ];
        }


        undo.capturedSquare =
            move.to;

        undo.capturedPiece =
            pos.board[
                move.to
            ];


        // Capture.
        if (
            move.enPassant
            ) {
            undo.capturedSquare =
                move.to +
                (
                    white
                    ? -8
                    : 8
                    );

            undo.capturedPiece =
                pos.board[
                    undo.capturedSquare
                ];

            removePiece(
                pos,
                undo.capturedSquare
            );
        }

        else if (
            undo.capturedPiece != '.'
            ) {
            removePiece(
                pos,
                move.to
            );
        }


        // Castling rights.
        if (
            movingPiece == 'K'
            ) {
            pos.castleWK = false;
            pos.castleWQ = false;
        }

        else if (
            movingPiece == 'k'
            ) {
            pos.castleBK = false;
            pos.castleBQ = false;
        }


        if (
            movingPiece == 'R'
            ) {
            if (
                move.from == 0
                ) {
                pos.castleWQ = false;
            }

            else if (
                move.from == 7
                ) {
                pos.castleWK = false;
            }
        }


        if (
            movingPiece == 'r'
            ) {
            if (
                move.from == 56
                ) {
                pos.castleBQ = false;
            }

            else if (
                move.from == 63
                ) {
                pos.castleBK = false;
            }
        }


        if (
            undo.capturedPiece == 'R'
            ) {
            if (
                undo.capturedSquare == 0
                ) {
                pos.castleWQ = false;
            }

            else if (
                undo.capturedSquare == 7
                ) {
                pos.castleWK = false;
            }
        }


        if (
            undo.capturedPiece == 'r'
            ) {
            if (
                undo.capturedSquare == 56
                ) {
                pos.castleBQ = false;
            }

            else if (
                undo.capturedSquare == 63
                ) {
                pos.castleBK = false;
            }
        }


        // Move main piece.
        removePiece(
            pos,
            move.from
        );

        char destinationPiece =
            movingPiece;

        if (
            move.promotion
            ) {
            destinationPiece =
                white
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
        }

        addPiece(
            pos,
            move.to,
            destinationPiece
        );


        // Castling rook.
        if (
            move.castle
            ) {
            if (
                movingPiece == 'K'
                ) {
                if (
                    move.to == 6
                    ) {
                    removePiece(
                        pos,
                        7
                    );

                    addPiece(
                        pos,
                        5,
                        'R'
                    );
                }

                else if (
                    move.to == 2
                    ) {
                    removePiece(
                        pos,
                        0
                    );

                    addPiece(
                        pos,
                        3,
                        'R'
                    );
                }
            }

            else if (
                movingPiece == 'k'
                ) {
                if (
                    move.to == 62
                    ) {
                    removePiece(
                        pos,
                        63
                    );

                    addPiece(
                        pos,
                        61,
                        'r'
                    );
                }

                else if (
                    move.to == 58
                    ) {
                    removePiece(
                        pos,
                        56
                    );

                    addPiece(
                        pos,
                        59,
                        'r'
                    );
                }
            }
        }


        // En passant target.
        pos.enPassantSquare =
            -1;

        if (
            std::tolower(
                static_cast<unsigned char>(
                    movingPiece
                    )
            )
            ==
            'p'
            &&
            std::abs(
                move.to -
                move.from
            )
            ==
            16
            ) {
            pos.enPassantSquare =
                (
                    move.from +
                    move.to
                    )
                /
                2;
        }


        // Clocks.
        if (
            std::tolower(
                static_cast<unsigned char>(
                    movingPiece
                    )
            )
            ==
            'p'
            ||
            undo.capturedPiece != '.'
            ) {
            pos.halfmoveClock =
                0;
        }

        else {
            ++pos.halfmoveClock;
        }

        if (
            !white
            ) {
            ++pos.fullmoveNumber;
        }


        // Side.
        pos.whiteToMove =
            !pos.whiteToMove;

        pos.zobristKey ^=
            ZOBRIST.side;


        // Add NEW castle + EP state.
        pos.zobristKey ^=
            ZOBRIST.castling[
                castlingIndex(pos)
            ];

        if (
            pos.enPassantSquare >= 0
            ) {
            pos.zobristKey ^=
                ZOBRIST.enPassantFile[
                    fileOf(
                        pos.enPassantSquare
                    )
                ];
        }
    }


    void makeMove(
        Position& pos,
        const Move& move
    ) {
        UndoState unused;

        makeMove(
            pos,
            move,
            unused
        );
    }


    // ============================================================
    // UNDO MOVE
    // ============================================================

    void undoMove(
        Position& pos,
        const Move& move,
        const UndoState& undo
    ) {
        pos.whiteToMove =
            !pos.whiteToMove;


        // Undo castling rook.
        if (
            move.castle
            ) {
            if (
                undo.movedPiece == 'K'
                ) {
                if (
                    move.to == 6
                    ) {
                    removePiece(
                        pos,
                        5
                    );

                    addPiece(
                        pos,
                        7,
                        'R'
                    );
                }

                else if (
                    move.to == 2
                    ) {
                    removePiece(
                        pos,
                        3
                    );

                    addPiece(
                        pos,
                        0,
                        'R'
                    );
                }
            }

            else if (
                undo.movedPiece == 'k'
                ) {
                if (
                    move.to == 62
                    ) {
                    removePiece(
                        pos,
                        61
                    );

                    addPiece(
                        pos,
                        63,
                        'r'
                    );
                }

                else if (
                    move.to == 58
                    ) {
                    removePiece(
                        pos,
                        59
                    );

                    addPiece(
                        pos,
                        56,
                        'r'
                    );
                }
            }
        }


        removePiece(
            pos,
            move.to
        );

        addPiece(
            pos,
            move.from,
            undo.movedPiece
        );


        if (
            undo.capturedPiece != '.'
            ) {
            addPiece(
                pos,
                undo.capturedSquare,
                undo.capturedPiece
            );
        }


        pos.castleWK =
            undo.castleWK;

        pos.castleWQ =
            undo.castleWQ;

        pos.castleBK =
            undo.castleBK;

        pos.castleBQ =
            undo.castleBQ;

        pos.enPassantSquare =
            undo.enPassantSquare;

        pos.halfmoveClock =
            undo.halfmoveClock;

        pos.fullmoveNumber =
            undo.fullmoveNumber;


        // The board/bitboards/occupancy have been restored.
        // Rather than reverse every hash operation manually,
        // restore the exact saved key.
        pos.zobristKey =
            undo.zobristKey;
    }


    // ============================================================
    // LEGAL MOVES
    // ============================================================

    void generateLegalMoves(
        Position& pos,
        MoveList& legal
    ) {
        legal.clear();

        MoveList pseudo;

        generatePseudoLegalMoves(
            pos,
            pseudo
        );

        const bool movingWhite =
            pos.whiteToMove;

        for (
            const Move& move :
            pseudo
            ) {
            UndoState undo;

            makeMove(
                pos,
                move,
                undo
            );

            if (
                !inCheck(
                    pos,
                    movingWhite
                )
                ) {
                legal.push_back(
                    move
                );
            }

            undoMove(
                pos,
                move,
                undo
            );
        }
    }


    MoveList generateLegalMoves(
        const Position& pos
    ) {
        Position work =
            pos;

        MoveList legal;

        generateLegalMoves(
            work,
            legal
        );

        return legal;
    }


    // ============================================================
    // PERFT
    // ============================================================

    namespace {

        std::uint64_t perftMutable(
            Position& pos,
            int depth
        ) {
            if (
                depth == 0
                ) {
                return 1;
            }

            MoveList moves;

            generateLegalMoves(
                pos,
                moves
            );

            if (
                depth == 1
                ) {
                return
                    static_cast<
                    std::uint64_t
                    >(
                        moves.count
                        );
            }

            std::uint64_t nodes = 0;

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

                nodes +=
                    perftMutable(
                        pos,
                        depth - 1
                    );

                undoMove(
                    pos,
                    move,
                    undo
                );
            }

            return nodes;
        }

    } // anonymous namespace


    std::uint64_t perft(
        const Position& pos,
        int depth
    ) {
        Position work =
            pos;

        return
            perftMutable(
                work,
                depth
            );
    }


    // ============================================================
    // NOTATION
    // ============================================================

    std::string squareName(
        int square
    ) {
        if (
            square < 0 ||
            square >= 64
            ) {
            return "??";
        }

        std::string name;

        name +=
            static_cast<char>(
                'a' +
                fileOf(square)
                );

        name +=
            static_cast<char>(
                '1' +
                rankOf(square)
                );

        return name;
    }


    std::string moveToUci(
        const Move& move
    ) {
        std::string result =
            squareName(
                move.from
            )
            +
            squareName(
                move.to
            );

        if (
            move.promotion
            ) {
            result +=
                static_cast<char>(
                    std::tolower(
                        static_cast<unsigned char>(
                            move.promotion
                            )
                    )
                    );
        }

        return result;
    }


    std::string moveToSan(
        const Position& pos,
        const Move& move
    ) {
        const char piece =
            pos.board[
                move.from
            ];

        if (
            piece == '.'
            ) {
            return "??";
        }

        const char type =
            static_cast<char>(
                std::tolower(
                    static_cast<unsigned char>(
                        piece
                        )
                )
                );

        std::string san;


        if (
            move.castle ||
            (
                type == 'k' &&
                std::abs(
                    fileOf(move.to) -
                    fileOf(move.from)
                )
                ==
                2
                )
            ) {
            san =
                fileOf(move.to) >
                fileOf(move.from)
                ? "O-O"
                : "O-O-O";
        }

        else {
            const bool capture =
                move.enPassant ||
                pos.board[
                    move.to
                ] !=
                '.';


                    if (
                        type != 'p'
                        ) {
                        san +=
                            static_cast<char>(
                                std::toupper(
                                    static_cast<unsigned char>(
                                        type
                                        )
                                )
                                );

                        const auto legal =
                            generateLegalMoves(
                                pos
                            );

                        bool conflict = false;
                        bool sameFile = false;
                        bool sameRank = false;

                        for (
                            const Move& other :
                            legal
                            ) {
                            if (
                                sameMove(
                                    move,
                                    other
                                )
                                ) {
                                continue;
                            }

                            if (
                                other.to !=
                                move.to
                                ) {
                                continue;
                            }

                            const char otherPiece =
                                pos.board[
                                    other.from
                                ];

                            if (
                                std::tolower(
                                    static_cast<unsigned char>(
                                        otherPiece
                                        )
                                )
                                !=
                                type
                                ) {
                                continue;
                            }

                            conflict = true;

                            if (
                                fileOf(
                                    other.from
                                )
                                ==
                                fileOf(
                                    move.from
                                )
                                ) {
                                sameFile = true;
                            }

                            if (
                                rankOf(
                                    other.from
                                )
                                ==
                                rankOf(
                                    move.from
                                )
                                ) {
                                sameRank = true;
                            }
                        }

                        if (
                            conflict
                            ) {
                            if (
                                !sameFile
                                ) {
                                san +=
                                    static_cast<char>(
                                        'a' +
                                        fileOf(
                                            move.from
                                        )
                                        );
                            }

                            else if (
                                !sameRank
                                ) {
                                san +=
                                    static_cast<char>(
                                        '1' +
                                        rankOf(
                                            move.from
                                        )
                                        );
                            }

                            else {
                                san +=
                                    squareName(
                                        move.from
                                    );
                            }
                        }
                    }

                    else if (
                        capture
                        ) {
                        san +=
                            static_cast<char>(
                                'a' +
                                fileOf(
                                    move.from
                                )
                                );
                    }


                    if (
                        capture
                        ) {
                        san += 'x';
                    }

                    san +=
                        squareName(
                            move.to
                        );


                    if (
                        move.promotion
                        ) {
                        san += '=';

                        san +=
                            static_cast<char>(
                                std::toupper(
                                    static_cast<unsigned char>(
                                        move.promotion
                                        )
                                )
                                );
                    }
        }


        Position next =
            pos;

        makeMove(
            next,
            move
        );

        if (
            inCheck(
                next,
                next.whiteToMove
            )
            ) {
            const auto replies =
                generateLegalMoves(
                    next
                );

            san +=
                replies.empty()
                ? '#'
                : '+';
        }

        return san;
    }

} // namespace chess