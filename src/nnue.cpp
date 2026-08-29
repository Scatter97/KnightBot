#include "nnue.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <immintrin.h>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace chess {

    namespace {

        // ========================================================
        // FILE MAGIC
        // ========================================================

        constexpr std::array<char, 8>
            NNUE_V1_MAGIC{
                'K',
                'N',
                'U',
                'E',
                '0',
                '0',
                '0',
                '1'
            };


        constexpr std::array<char, 8>
            NNUE_V2_MAGIC{
                'K',
                'N',
                'U',
                'E',
                '0',
                '0',
                '0',
                '2'
            };


        // ========================================================
        // NETWORK TYPE
        // ========================================================

        enum class NetworkFormat {
            None = 0,
            V1 = 1,
            HalfKP512 = 2
        };


        // ========================================================
        // V1 NETWORK
        // ========================================================

        struct NetworkV1 {

            std::array<
                std::int32_t,
                NNUE_V1_HIDDEN_COUNT
            > hiddenBias{};


            // Feature-major:
            //
            // inputWeights[
            //      feature * hiddenCount +
            //      hidden
            // ]
            std::vector<std::int16_t>
                inputWeights;


            std::array<
                std::int16_t,
                NNUE_V1_HIDDEN_COUNT
            > outputWeights{};


            std::int32_t outputBias =
                0;


            std::int32_t activationMax =
                127;


            std::int32_t outputScale =
                64;
        };


        // ========================================================
        // HALFKP-512 NETWORK
        // ========================================================

        struct HalfKP512Network {

            std::array<
                std::int32_t,
                HALFKP_TRANSFORMER_HIDDEN
            > transformerBias{};


            // Feature-major:
            //
            // transformerWeights[
            //      feature * 512 +
            //      hidden
            // ]
            std::vector<std::int16_t>
                transformerWeights;


            std::array<
                std::int32_t,
                HALFKP_DENSE1_COUNT
            > dense1Bias{};


            // Output-major:
            //
            // dense1Weights[
            //      output * HALFKP_DENSE1_INPUT_COUNT +
            //      input
            // ]
            std::vector<std::int16_t>
                dense1Weights;


            std::array<
                std::int32_t,
                HALFKP_DENSE2_COUNT
            > dense2Bias{};


            // Output-major:
            //
            // dense2Weights[
            //      output * HALFKP_DENSE1_COUNT +
            //      input
            // ]
            std::vector<std::int16_t>
                dense2Weights;


            std::array<
                std::int16_t,
                HALFKP_DENSE2_COUNT
            > outputWeights{};


            std::int32_t outputBias =
                0;


            std::int32_t activationMax =
                0;


            std::int32_t activationScale =
                0;


            std::int32_t weightScale =
                0;


            std::int32_t outputScale =
                0;


            std::int32_t contextActive =
                0;
        };


        // ========================================================
        // ACTIVE NETWORK
        // ========================================================

        struct Network {

            NetworkFormat format =
                NetworkFormat::None;


            NetworkV1 v1;


            HalfKP512Network halfKP;


            bool loaded =
                false;


            std::string sourceFile;
        };


        Network network;


        bool useNNUE =
            false;

        std::uint64_t networkGeneration =
            1;
        // ========================================================
        // BINARY IO
        // ========================================================

        template<typename T>
        bool readValue(
            std::ifstream& input,
            T& value
        ) {
            input.read(
                reinterpret_cast<char*>(
                    &value
                ),
                sizeof(T)
            );


            return
                static_cast<bool>(
                    input
                );
        }


        template<typename T>
        bool writeValue(
            std::ofstream& output,
            const T& value
        ) {
            output.write(
                reinterpret_cast<
                    const char*
                >(
                    &value
                ),
                sizeof(T)
            );


            return
                static_cast<bool>(
                    output
                );
        }


        template<typename T>
        bool readArray(
            std::ifstream& input,
            T* destination,
            std::size_t count
        ) {
            if (
                count == 0
            ) {
                return true;
            }


            const std::size_t bytes =
                sizeof(T) *
                count;


            if (
                bytes >
                static_cast<std::size_t>(
                    std::numeric_limits<
                        std::streamsize
                    >::max()
                )
            ) {
                return false;
            }


            input.read(
                reinterpret_cast<char*>(
                    destination
                ),
                static_cast<
                    std::streamsize
                >(
                    bytes
                )
            );


            return
                static_cast<bool>(
                    input
                );
        }


        bool fileHasTrailingData(
            std::ifstream& input
        ) {
            char extra =
                0;


            input.read(
                &extra,
                1
            );


            return
                input.gcount() !=
                0;
        }


        // ========================================================
        // PIECE HELPERS
        // ========================================================

        int pieceIndexForChar(
            char piece
        ) {
            switch (
                piece
                ) {
            case 'P':
                return WP;

            case 'N':
                return WN;

            case 'B':
                return WB;

            case 'R':
                return WR;

            case 'Q':
                return WQ;

            case 'K':
                return WK;


            case 'p':
                return BP;

            case 'n':
                return BN;

            case 'b':
                return BB;

            case 'r':
                return BR;

            case 'q':
                return BQ;

            case 'k':
                return BK;

            default:
                return -1;
            }
        }


        bool pieceIsWhite(
            char piece
        ) {
            return
                piece >= 'A' &&
                piece <= 'Z';
        }


        bool pieceIsKing(
            char piece
        ) {
            return
                piece == 'K' ||
                piece == 'k';
        }


        int halfKPPieceTypeIndex(
            char piece
        ) {
            switch (
                piece
                ) {
            case 'P':
            case 'p':
                return 0;

            case 'N':
            case 'n':
                return 1;

            case 'B':
            case 'b':
                return 2;

            case 'R':
            case 'r':
                return 3;

            case 'Q':
            case 'q':
                return 4;

            default:
                return -1;
            }
        }


        int orientSquare(
            bool perspectiveWhite,
            int square
        ) {
            if (
                perspectiveWhite
            ) {
                return square;
            }


            return
                square ^
                56;
        }


        int findKingSquare(
            const Position& pos,
            bool whiteKing
        ) {
            const char king =
                whiteKing
                ? 'K'
                : 'k';


            for (
                int square = 0;
                square < 64;
                ++square
                ) {
                if (
                    pos.board[
                        square
                    ] ==
                    king
                ) {
                    return square;
                }
            }


            return -1;
        }


        // ========================================================
        // V1 MATERIAL VALUE FOR COMPATIBILITY NETWORK
        // ========================================================

        int compatibilityMaterialValue(
            int pieceIndex
        ) {
            switch (
                pieceIndex
                ) {
            case WP:
            case BP:
                return 100;

            case WN:
            case BN:
                return 320;

            case WB:
            case BB:
                return 330;

            case WR:
            case BR:
                return 500;

            case WQ:
            case BQ:
                return 900;

            case WK:
            case BK:
                return 0;

            default:
                return 0;
            }
        }


        // ========================================================
        // CONTEXT FEATURES
        // ========================================================
        //
        // Python layout:
        //
        // bit 0  White to move
        // bit 1  White kingside castling
        // bit 2  White queenside castling
        // bit 3  Black kingside castling
        // bit 4  Black queenside castling
        //
        // bits 5..12:
        //
        //      en-passant file a..h
        //
        // ========================================================

        std::uint16_t contextBits(
            const Position& pos
        ) {
            std::uint16_t bits =
                0;


            if (
                pos.whiteToMove
            ) {
                bits |=
                    static_cast<
                        std::uint16_t
                    >(
                        1u << 0
                    );
            }


            if (
                pos.castleWK
            ) {
                bits |=
                    static_cast<
                        std::uint16_t
                    >(
                        1u << 1
                    );
            }


            if (
                pos.castleWQ
            ) {
                bits |=
                    static_cast<
                        std::uint16_t
                    >(
                        1u << 2
                    );
            }


            if (
                pos.castleBK
            ) {
                bits |=
                    static_cast<
                        std::uint16_t
                    >(
                        1u << 3
                    );
            }


            if (
                pos.castleBQ
            ) {
                bits |=
                    static_cast<
                        std::uint16_t
                    >(
                        1u << 4
                    );
            }


            if (
                pos.enPassantSquare >= 0 &&
                pos.enPassantSquare < 64
            ) {
                const int file =
                    pos.enPassantSquare %
                    8;


                bits |=
                    static_cast<
                        std::uint16_t
                    >(
                        1u <<
                        (
                            5 +
                            file
                        )
                    );
            }


            return bits;
        }


        // ========================================================
        // SIGNED INTEGER DIVISION
        // ========================================================
        //
        // Python integer HalfKP inference explicitly truncates
        // toward zero.
        //
        // C++ signed integer division also truncates toward zero.
        //
        // Keeping this helper makes that requirement explicit.
        // ========================================================

        std::int64_t divideTowardZero(
            std::int64_t value,
            std::int64_t divisor
        ) {
            return
                value /
                divisor;
        }


        // ========================================================
        // LOAD V1
        // ========================================================

        bool loadV1(
            std::ifstream& input,
            const std::string& path,
            Network& candidate,
            std::string* errorMessage
        ) {
            auto fail =
                [
                    errorMessage
                ](
                    const std::string& message
                    ) {
                    if (
                        errorMessage !=
                        nullptr
                    ) {
                        *errorMessage =
                            message;
                    }


                    return false;
                };


            std::uint32_t version =
                0;

            std::uint32_t inputCount =
                0;

            std::uint32_t hiddenCount =
                0;

            std::int32_t activationMax =
                0;

            std::int32_t outputScale =
                0;


            if (
                !readValue(
                    input,
                    version
                )
                ||
                !readValue(
                    input,
                    inputCount
                )
                ||
                !readValue(
                    input,
                    hiddenCount
                )
                ||
                !readValue(
                    input,
                    activationMax
                )
                ||
                !readValue(
                    input,
                    outputScale
                )
            ) {
                return
                    fail(
                        "NNUE V1 file ended while reading header."
                    );
            }


            if (
                version !=
                NNUE_V1_FORMAT_VERSION
            ) {
                return
                    fail(
                        "Unsupported KnightBot NNUE V1 format version."
                    );
            }


            if (
                inputCount !=
                static_cast<
                    std::uint32_t
                >(
                    NNUE_V1_INPUT_COUNT
                )
            ) {
                return
                    fail(
                        "NNUE V1 input count does not match KnightBot."
                    );
            }


            if (
                hiddenCount !=
                static_cast<
                    std::uint32_t
                >(
                    NNUE_V1_HIDDEN_COUNT
                )
            ) {
                return
                    fail(
                        "NNUE V1 hidden size does not match KnightBot."
                    );
            }


            if (
                activationMax <= 0
            ) {
                return
                    fail(
                        "NNUE V1 activationMax must be positive."
                    );
            }


            if (
                outputScale <= 0
            ) {
                return
                    fail(
                        "NNUE V1 outputScale must be positive."
                    );
            }


            NetworkV1& v1 =
                candidate.v1;


            v1.activationMax =
                activationMax;

            v1.outputScale =
                outputScale;


            if (
                !readArray(
                    input,
                    v1.hiddenBias.data(),
                    v1.hiddenBias.size()
                )
            ) {
                return
                    fail(
                        "NNUE V1 ended while reading hidden biases."
                    );
            }


            const std::size_t
                inputWeightCount =
                    static_cast<
                        std::size_t
                    >(
                        NNUE_V1_INPUT_COUNT
                    )
                    *
                    NNUE_V1_HIDDEN_COUNT;


            v1.inputWeights.resize(
                inputWeightCount
            );


            if (
                !readArray(
                    input,
                    v1.inputWeights.data(),
                    v1.inputWeights.size()
                )
            ) {
                return
                    fail(
                        "NNUE V1 ended while reading input weights."
                    );
            }


            if (
                !readArray(
                    input,
                    v1.outputWeights.data(),
                    v1.outputWeights.size()
                )
            ) {
                return
                    fail(
                        "NNUE V1 ended while reading output weights."
                    );
            }


            if (
                !readValue(
                    input,
                    v1.outputBias
                )
            ) {
                return
                    fail(
                        "NNUE V1 ended while reading output bias."
                    );
            }


            if (
                fileHasTrailingData(
                    input
                )
            ) {
                return
                    fail(
                        "Unexpected trailing data in NNUE V1 file."
                    );
            }


            candidate.format =
                NetworkFormat::V1;

            candidate.loaded =
                true;

            candidate.sourceFile =
                path;


            return true;
        }


        // ========================================================
        // LOAD HALFKP-512
        // ========================================================

        bool loadHalfKP512(
            std::ifstream& input,
            const std::string& path,
            Network& candidate,
            std::string* errorMessage
        ) {
            auto fail =
                [
                    errorMessage
                ](
                    const std::string& message
                    ) {
                    if (
                        errorMessage !=
                        nullptr
                    ) {
                        *errorMessage =
                            message;
                    }


                    return false;
                };


            std::uint32_t version =
                0;

            std::uint32_t featureCount =
                0;

            std::uint32_t transformerHidden =
                0;

            std::uint32_t contextCount =
                0;

            std::uint32_t dense1Count =
                0;

            std::uint32_t dense2Count =
                0;


            std::int32_t activationMax =
                0;

            std::int32_t activationScale =
                0;

            std::int32_t weightScale =
                0;

            std::int32_t outputScale =
                0;

            std::int32_t contextActive =
                0;


            if (
                !readValue(
                    input,
                    version
                )
                ||
                !readValue(
                    input,
                    featureCount
                )
                ||
                !readValue(
                    input,
                    transformerHidden
                )
                ||
                !readValue(
                    input,
                    contextCount
                )
                ||
                !readValue(
                    input,
                    dense1Count
                )
                ||
                !readValue(
                    input,
                    dense2Count
                )
                ||
                !readValue(
                    input,
                    activationMax
                )
                ||
                !readValue(
                    input,
                    activationScale
                )
                ||
                !readValue(
                    input,
                    weightScale
                )
                ||
                !readValue(
                    input,
                    outputScale
                )
                ||
                !readValue(
                    input,
                    contextActive
                )
            ) {
                return
                    fail(
                        "HalfKP-512 file ended while reading header."
                    );
            }


            if (
                version !=
                HALFKP_FORMAT_VERSION
            ) {
                return
                    fail(
                        "Unsupported HalfKP-512 format version."
                    );
            }


            if (
                featureCount !=
                static_cast<
                    std::uint32_t
                >(
                    HALFKP_FEATURE_COUNT
                )
            ) {
                return
                    fail(
                        "HalfKP-512 feature count mismatch."
                    );
            }


            if (
                transformerHidden !=
                static_cast<
                    std::uint32_t
                >(
                    HALFKP_TRANSFORMER_HIDDEN
                )
            ) {
                return
                    fail(
                        "HalfKP-512 transformer size mismatch."
                    );
            }


            if (
                contextCount !=
                static_cast<
                    std::uint32_t
                >(
                    HALFKP_CONTEXT_COUNT
                )
            ) {
                return
                    fail(
                        "HalfKP-512 context count mismatch."
                    );
            }


            if (
                dense1Count !=
                static_cast<
                    std::uint32_t
                >(
                    HALFKP_DENSE1_COUNT
                )
            ) {
                return
                    fail(
                        "HalfKP-512 dense1 size mismatch."
                    );
            }


            if (
                dense2Count !=
                static_cast<
                    std::uint32_t
                >(
                    HALFKP_DENSE2_COUNT
                )
            ) {
                return
                    fail(
                        "HalfKP-512 dense2 size mismatch."
                    );
            }


            if (
                activationMax <= 0 ||
                activationScale <= 0 ||
                weightScale <= 0 ||
                outputScale <= 0 ||
                contextActive <= 0
            ) {
                return
                    fail(
                        "HalfKP-512 contains invalid quantization scales."
                    );
            }


            HalfKP512Network& halfKP =
                candidate.halfKP;


            halfKP.activationMax =
                activationMax;

            halfKP.activationScale =
                activationScale;

            halfKP.weightScale =
                weightScale;

            halfKP.outputScale =
                outputScale;

            halfKP.contextActive =
                contextActive;


            // ====================================================
            // TRANSFORMER BIAS
            // ====================================================

            if (
                !readArray(
                    input,
                    halfKP.transformerBias.data(),
                    halfKP.transformerBias.size()
                )
            ) {
                return
                    fail(
                        "HalfKP-512 ended while reading "
                        "transformer biases."
                    );
            }


            // ====================================================
            // TRANSFORMER WEIGHTS
            // ====================================================

            const std::size_t
                transformerWeightCount =
                    static_cast<
                        std::size_t
                    >(
                        HALFKP_FEATURE_COUNT
                    )
                    *
                    HALFKP_TRANSFORMER_HIDDEN;


            halfKP.transformerWeights.resize(
                transformerWeightCount
            );


            if (
                !readArray(
                    input,
                    halfKP.transformerWeights.data(),
                    halfKP.transformerWeights.size()
                )
            ) {
                return
                    fail(
                        "HalfKP-512 ended while reading "
                        "transformer weights."
                    );
            }


            // ====================================================
            // DENSE 1 BIAS
            // ====================================================

            if (
                !readArray(
                    input,
                    halfKP.dense1Bias.data(),
                    halfKP.dense1Bias.size()
                )
            ) {
                return
                    fail(
                        "HalfKP-512 ended while reading dense1 biases."
                    );
            }


            // ====================================================
            // DENSE 1 WEIGHTS
            // ====================================================

            const std::size_t
                dense1WeightCount =
                    static_cast<
                        std::size_t
                    >(
                        HALFKP_DENSE1_COUNT
                    )
                    *
                    HALFKP_DENSE1_INPUT_COUNT;


            halfKP.dense1Weights.resize(
                dense1WeightCount
            );


            if (
                !readArray(
                    input,
                    halfKP.dense1Weights.data(),
                    halfKP.dense1Weights.size()
                )
            ) {
                return
                    fail(
                        "HalfKP-512 ended while reading dense1 weights."
                    );
            }


            // ====================================================
            // DENSE 2 BIAS
            // ====================================================

            if (
                !readArray(
                    input,
                    halfKP.dense2Bias.data(),
                    halfKP.dense2Bias.size()
                )
            ) {
                return
                    fail(
                        "HalfKP-512 ended while reading dense2 biases."
                    );
            }


            // ====================================================
            // DENSE 2 WEIGHTS
            // ====================================================

            const std::size_t
                dense2WeightCount =
                    static_cast<
                        std::size_t
                    >(
                        HALFKP_DENSE2_COUNT
                    )
                    *
                    HALFKP_DENSE1_COUNT;


            halfKP.dense2Weights.resize(
                dense2WeightCount
            );


            if (
                !readArray(
                    input,
                    halfKP.dense2Weights.data(),
                    halfKP.dense2Weights.size()
                )
            ) {
                return
                    fail(
                        "HalfKP-512 ended while reading dense2 weights."
                    );
            }


            // ====================================================
            // OUTPUT WEIGHTS
            // ====================================================

            if (
                !readArray(
                    input,
                    halfKP.outputWeights.data(),
                    halfKP.outputWeights.size()
                )
            ) {
                return
                    fail(
                        "HalfKP-512 ended while reading output weights."
                    );
            }


            // ====================================================
            // OUTPUT BIAS
            // ====================================================

            if (
                !readValue(
                    input,
                    halfKP.outputBias
                )
            ) {
                return
                    fail(
                        "HalfKP-512 ended while reading output bias."
                    );
            }


            if (
                fileHasTrailingData(
                    input
                )
            ) {
                return
                    fail(
                        "Unexpected trailing data in HalfKP-512 file."
                    );
            }


            candidate.format =
                NetworkFormat::HalfKP512;

            candidate.loaded =
                true;

            candidate.sourceFile =
                path;


            return true;
        }


        // ========================================================
        // V1 EVALUATION
        // ========================================================

        int evaluateV1(
            const Position& pos
        ) {
            const NetworkV1& v1 =
                network.v1;


            std::array<
                std::int32_t,
                NNUE_V1_HIDDEN_COUNT
            > accumulator =
                v1.hiddenBias;


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


                const int feature =
                    nnueFeatureIndex(
                        piece,
                        square
                    );


                if (
                    feature < 0
                ) {
                    continue;
                }


                const std::size_t base =
                    static_cast<
                        std::size_t
                    >(
                        feature
                    )
                    *
                    NNUE_V1_HIDDEN_COUNT;


                for (
                    int hidden = 0;
                    hidden <
                    NNUE_V1_HIDDEN_COUNT;
                    ++hidden
                    ) {
                    accumulator[
                        hidden
                    ] +=
                        v1.inputWeights[
                            base +
                            static_cast<
                                std::size_t
                            >(
                                hidden
                            )
                        ];
                }
            }


            std::int64_t output =
                v1.outputBias;


            for (
                int hidden = 0;
                hidden <
                NNUE_V1_HIDDEN_COUNT;
                ++hidden
                ) {
                const std::int32_t
                    activation =
                        std::clamp(
                            accumulator[
                                hidden
                            ],
                            0,
                            v1.activationMax
                        );


                output +=
                    static_cast<
                        std::int64_t
                    >(
                        activation
                    )
                    *
                    static_cast<
                        std::int64_t
                    >(
                        v1.outputWeights[
                            hidden
                        ]
                    );
            }


            output =
                divideTowardZero(
                    output,
                    v1.outputScale
                );


            output =
                std::clamp<
                    std::int64_t
                >(
                    output,
                    -30000,
                    30000
                );


            return
                static_cast<int>(
                    output
                );
        }


        // ========================================================
        // BUILD HALFKP ACCUMULATOR
        // ========================================================

        bool buildHalfKPAccumulator(
            const Position& pos,
            bool perspectiveWhite,
            int kingSquare,
            std::array<
                std::int64_t,
                HALFKP_TRANSFORMER_HIDDEN
            >& accumulator
        ) {
            const HalfKP512Network& halfKP =
                network.halfKP;


            for (
                int hidden = 0;
                hidden <
                HALFKP_TRANSFORMER_HIDDEN;
                ++hidden
                ) {
                accumulator[
                    hidden
                ] =
                    halfKP.transformerBias[
                        hidden
                    ];
            }


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
                    piece == '.' ||
                    pieceIsKing(
                        piece
                    )
                ) {
                    continue;
                }


                const int feature =
                    halfKPFeatureIndex(
                        perspectiveWhite,
                        kingSquare,
                        piece,
                        square
                    );


                if (
                    feature < 0
                ) {
                    return false;
                }


                const std::size_t base =
                    static_cast<
                        std::size_t
                    >(
                        feature
                    )
                    *
                    HALFKP_TRANSFORMER_HIDDEN;


                for (
                    int hidden = 0;
                    hidden <
                    HALFKP_TRANSFORMER_HIDDEN;
                    ++hidden
                    ) {
                    accumulator[
                        hidden
                    ] +=
                        static_cast<
                            std::int64_t
                        >(
                            halfKP.transformerWeights[
                                base +
                                static_cast<
                                    std::size_t
                                >(
                                    hidden
                                )
                            ]
                        );
                }
            }


            return true;
        }

        // ========================================================
        // HALFKP CACHE HELPERS
        // ========================================================

        bool halfKPCacheUsable(
            const Position& pos
        ) {
            return
                network.loaded
                &&
                network.format ==
                NetworkFormat::HalfKP512
                &&
                pos.halfKPValid
                &&
                pos.halfKPGeneration ==
                networkGeneration;
        }


        void copyAccumulatorToPosition(
            const std::array<
                std::int64_t,
                HALFKP_TRANSFORMER_HIDDEN
            >& source,
            std::array<
                std::int32_t,
                HALFKP_TRANSFORMER_HIDDEN
            >& destination
        ) {
            for (
                int hidden = 0;
                hidden <
                HALFKP_TRANSFORMER_HIDDEN;
                ++hidden
                ) {
                destination[
                    hidden
                ] =
                    static_cast<
                        std::int32_t
                    >(
                        source[
                            hidden
                        ]
                    );
            }
        }


        bool rebuildPerspective(
            Position& pos,
            bool perspectiveWhite
        ) {
            const int kingSquare =
                findKingSquare(
                    pos,
                    perspectiveWhite
                );


            if (
                kingSquare < 0
            ) {
                pos.halfKPValid =
                    false;

                return false;
            }


            std::array<
                std::int64_t,
                HALFKP_TRANSFORMER_HIDDEN
            > accumulator{};


            if (
                !buildHalfKPAccumulator(
                    pos,
                    perspectiveWhite,
                    kingSquare,
                    accumulator
                )
            ) {
                pos.halfKPValid =
                    false;

                return false;
            }


            if (
                perspectiveWhite
            ) {
                copyAccumulatorToPosition(
                    accumulator,
                    pos.halfKPWhiteAccumulator
                );

                pos.halfKPWhiteKingSquare =
                    kingSquare;
            }
            else {
                copyAccumulatorToPosition(
                    accumulator,
                    pos.halfKPBlackAccumulator
                );

                pos.halfKPBlackKingSquare =
                    kingSquare;
            }


            return true;
        }


        void applyFeatureDelta(
            Position& pos,
            bool perspectiveWhite,
            char piece,
            int square,
            int sign
        ) {
            if (
                piece == '.'
                ||
                pieceIsKing(
                    piece
                )
            ) {
                return;
            }


            const int kingSquare =
                perspectiveWhite
                ? pos.halfKPWhiteKingSquare
                : pos.halfKPBlackKingSquare;


            const int feature =
                halfKPFeatureIndex(
                    perspectiveWhite,
                    kingSquare,
                    piece,
                    square
                );


            if (
                feature < 0
            ) {
                pos.halfKPValid =
                    false;

                return;
            }


            const std::size_t base =
                static_cast<std::size_t>(
                    feature
                )
                *
                HALFKP_TRANSFORMER_HIDDEN;


            auto& accumulator =
                perspectiveWhite
                ? pos.halfKPWhiteAccumulator
                : pos.halfKPBlackAccumulator;


            const std::int16_t* weights =
                network.halfKP.transformerWeights.data()
                +
                base;


            int hidden =
                0;


            // ====================================================
            // AVX2
            //
            // 16 int16 weights are loaded at once.
            // They are widened into two groups of 8 int32 values
            // because the accumulator itself is int32.
            // ====================================================

            const bool subtract =
                sign < 0;


            for (
                ;
                hidden + 16 <=
                HALFKP_TRANSFORMER_HIDDEN;
                hidden += 16
                ) {
                const __m256i packedWeights =
                    _mm256_loadu_si256(
                        reinterpret_cast<
                            const __m256i*
                        >(
                            weights +
                            hidden
                        )
                    );


                const __m128i low16 =
                    _mm256_castsi256_si128(
                        packedWeights
                    );


                const __m128i high16 =
                    _mm256_extracti128_si256(
                        packedWeights,
                        1
                    );


                __m256i weightLow32 =
                    _mm256_cvtepi16_epi32(
                        low16
                    );


                __m256i weightHigh32 =
                    _mm256_cvtepi16_epi32(
                        high16
                    );


                if (
                    subtract
                    ) {
                    weightLow32 =
                        _mm256_sub_epi32(
                            _mm256_setzero_si256(),
                            weightLow32
                        );


                    weightHigh32 =
                        _mm256_sub_epi32(
                            _mm256_setzero_si256(),
                            weightHigh32
                        );
                }


                __m256i accumulatorLow =
                    _mm256_loadu_si256(
                        reinterpret_cast<
                            const __m256i*
                        >(
                            accumulator.data() +
                            hidden
                        )
                    );


                __m256i accumulatorHigh =
                    _mm256_loadu_si256(
                        reinterpret_cast<
                            const __m256i*
                        >(
                            accumulator.data() +
                            hidden +
                            8
                        )
                    );


                accumulatorLow =
                    _mm256_add_epi32(
                        accumulatorLow,
                        weightLow32
                    );


                accumulatorHigh =
                    _mm256_add_epi32(
                        accumulatorHigh,
                        weightHigh32
                    );


                _mm256_storeu_si256(
                    reinterpret_cast<
                        __m256i*
                    >(
                        accumulator.data() +
                        hidden
                    ),
                    accumulatorLow
                );


                _mm256_storeu_si256(
                    reinterpret_cast<
                        __m256i*
                    >(
                        accumulator.data() +
                        hidden +
                        8
                    ),
                    accumulatorHigh
                );
            }


            // Scalar tail.
            //
            // 512 is divisible by 16, so HalfKP-512 currently
            // doesn't need this, but keeping it makes the function
            // robust if the transformer size ever changes.
            for (
                ;
                hidden <
                HALFKP_TRANSFORMER_HIDDEN;
                ++hidden
                ) {
                accumulator[
                    hidden
                ] +=
                    sign
                    *
                    static_cast<std::int32_t>(
                        weights[
                            hidden
                        ]
                    );
            }
        }

        char destinationPieceForMove(
            const Move& move,
            const UndoState& undo
        ) {
            char result =
                undo.movedPiece;


            if (
                !move.promotion
            ) {
                return result;
            }


            const bool white =
                pieceIsWhite(
                    undo.movedPiece
                );


            switch (
                move.promotion
                ) {
            case 'q':
            case 'Q':
                return
                    white
                    ? 'Q'
                    : 'q';

            case 'r':
            case 'R':
                return
                    white
                    ? 'R'
                    : 'r';

            case 'b':
            case 'B':
                return
                    white
                    ? 'B'
                    : 'b';

            case 'n':
            case 'N':
                return
                    white
                    ? 'N'
                    : 'n';

            default:
                return result;
            }
        }


        void applyCastlingRookForward(
            Position& pos,
            bool perspectiveWhite,
            const Move& move,
            char movingPiece
        ) {
            if (
                !move.castle
            ) {
                return;
            }


            if (
                movingPiece == 'K'
            ) {
                if (
                    move.to == 6
                ) {
                    applyFeatureDelta(
                        pos,
                        perspectiveWhite,
                        'R',
                        7,
                        -1
                    );

                    applyFeatureDelta(
                        pos,
                        perspectiveWhite,
                        'R',
                        5,
                        +1
                    );
                }
                else if (
                    move.to == 2
                ) {
                    applyFeatureDelta(
                        pos,
                        perspectiveWhite,
                        'R',
                        0,
                        -1
                    );

                    applyFeatureDelta(
                        pos,
                        perspectiveWhite,
                        'R',
                        3,
                        +1
                    );
                }
            }
            else if (
                movingPiece == 'k'
            ) {
                if (
                    move.to == 62
                ) {
                    applyFeatureDelta(
                        pos,
                        perspectiveWhite,
                        'r',
                        63,
                        -1
                    );

                    applyFeatureDelta(
                        pos,
                        perspectiveWhite,
                        'r',
                        61,
                        +1
                    );
                }
                else if (
                    move.to == 58
                ) {
                    applyFeatureDelta(
                        pos,
                        perspectiveWhite,
                        'r',
                        56,
                        -1
                    );

                    applyFeatureDelta(
                        pos,
                        perspectiveWhite,
                        'r',
                        59,
                        +1
                    );
                }
            }
        }


        void applyCastlingRookBackward(
            Position& pos,
            bool perspectiveWhite,
            const Move& move,
            char movingPiece
        ) {
            if (
                !move.castle
            ) {
                return;
            }


            if (
                movingPiece == 'K'
            ) {
                if (
                    move.to == 6
                ) {
                    applyFeatureDelta(
                        pos,
                        perspectiveWhite,
                        'R',
                        5,
                        -1
                    );

                    applyFeatureDelta(
                        pos,
                        perspectiveWhite,
                        'R',
                        7,
                        +1
                    );
                }
                else if (
                    move.to == 2
                ) {
                    applyFeatureDelta(
                        pos,
                        perspectiveWhite,
                        'R',
                        3,
                        -1
                    );

                    applyFeatureDelta(
                        pos,
                        perspectiveWhite,
                        'R',
                        0,
                        +1
                    );
                }
            }
            else if (
                movingPiece == 'k'
            ) {
                if (
                    move.to == 62
                ) {
                    applyFeatureDelta(
                        pos,
                        perspectiveWhite,
                        'r',
                        61,
                        -1
                    );

                    applyFeatureDelta(
                        pos,
                        perspectiveWhite,
                        'r',
                        63,
                        +1
                    );
                }
                else if (
                    move.to == 58
                ) {
                    applyFeatureDelta(
                        pos,
                        perspectiveWhite,
                        'r',
                        59,
                        -1
                    );

                    applyFeatureDelta(
                        pos,
                        perspectiveWhite,
                        'r',
                        56,
                        +1
                    );
                }
            }
        }

        // ========================================================
        // HALFKP-512 EVALUATION
        // ========================================================

                int evaluateHalfKP512(
            const Position& constPos
        ) {
            const HalfKP512Network& halfKP =
                network.halfKP;


            // The cache is logically mutable evaluation state.
            Position& pos =
                const_cast<Position&>(
                    constPos
                );


            if (
                !halfKPCacheUsable(
                    pos
                )
            ) {
                rebuildHalfKPAccumulators(
                    pos
                );
            }


            if (
                !halfKPCacheUsable(
                    pos
                )
            ) {
                return 0;
            }


            std::array<
                std::int32_t,
                HALFKP_DENSE1_INPUT_COUNT
            > combined{};


            for (
                int hidden = 0;
                hidden <
                HALFKP_TRANSFORMER_HIDDEN;
                ++hidden
                ) {
                combined[
                    hidden
                ] =
                    std::clamp<
                        std::int64_t
                    >(
                        pos.halfKPWhiteAccumulator[
                            hidden
                        ],
                        0,
                        halfKP.activationMax
                    );


                combined[
                    HALFKP_TRANSFORMER_HIDDEN +
                    hidden
                ] =
                    std::clamp<
                        std::int64_t
                    >(
                        pos.halfKPBlackAccumulator[
                            hidden
                        ],
                        0,
                        halfKP.activationMax
                    );
            }


            const std::uint16_t bits =
                contextBits(
                    pos
                );


            const int contextBase =
                HALFKP_TRANSFORMER_HIDDEN *
                2;


            for (
                int contextIndex = 0;
                contextIndex <
                HALFKP_CONTEXT_COUNT;
                ++contextIndex
                ) {
                combined[
                    contextBase +
                    contextIndex
                ] =
                    (
                        bits &
                        static_cast<
                            std::uint16_t
                        >(
                            1u <<
                            contextIndex
                        )
                    )
                    ?
                    halfKP.contextActive
                    :
                    0;
            }


            std::array<
                std::int64_t,
                HALFKP_DENSE1_COUNT
            > hidden1{};


            for (
                int outputIndex = 0;
                outputIndex <
                HALFKP_DENSE1_COUNT;
                ++outputIndex
                ) {
                std::int64_t raw =
                    halfKP.dense1Bias[
                        outputIndex
                    ];


                const std::size_t base =
                    static_cast<
                        std::size_t
                    >(
                        outputIndex
                    )
                    *
                    HALFKP_DENSE1_INPUT_COUNT;


                const std::int16_t* weights =
                    halfKP.dense1Weights.data()
                    +
                    base;


                int inputIndex =
                    0;


                // ================================================
                // AVX2 DENSE-1 DOT PRODUCT
                //
                // 8 x int32 inputs
                // 8 x int16 weights -> widened to int32
                //
                // Products are accumulated as int64 so the result
                // remains exactly equivalent to the scalar code.
                // ================================================

                __m256i sumEven =
                    _mm256_setzero_si256();

                __m256i sumOdd =
                    _mm256_setzero_si256();


                for (
                    ;
                    inputIndex + 8 <=
                    HALFKP_DENSE1_INPUT_COUNT;
                    inputIndex += 8
                    ) {
                    const __m256i inputs =
                        _mm256_loadu_si256(
                            reinterpret_cast<
                                const __m256i*
                            >(
                                combined.data() +
                                inputIndex
                            )
                        );


                    const __m128i packedWeights =
                        _mm_loadu_si128(
                            reinterpret_cast<
                                const __m128i*
                            >(
                                weights +
                                inputIndex
                            )
                        );


                    const __m256i weights32 =
                        _mm256_cvtepi16_epi32(
                            packedWeights
                        );


                    // _mm256_mul_epi32 multiplies lanes
                    // 0,2,4,6 and produces four int64 results.
                    const __m256i productsEven =
                        _mm256_mul_epi32(
                            inputs,
                            weights32
                        );


                    // Move odd 32-bit lanes into even positions,
                    // then multiply those as signed int32 values.
                    const __m256i inputsOdd =
                        _mm256_srli_epi64(
                            inputs,
                            32
                        );


                    const __m256i weightsOdd =
                        _mm256_srli_epi64(
                            weights32,
                            32
                        );


                    const __m256i productsOdd =
                        _mm256_mul_epi32(
                            inputsOdd,
                            weightsOdd
                        );


                    sumEven =
                        _mm256_add_epi64(
                            sumEven,
                            productsEven
                        );


                    sumOdd =
                        _mm256_add_epi64(
                            sumOdd,
                            productsOdd
                        );
                }


                alignas(32)
                std::int64_t partialEven[
                    4
                ]{};


                alignas(32)
                std::int64_t partialOdd[
                    4
                ]{};


                _mm256_store_si256(
                    reinterpret_cast<
                        __m256i*
                    >(
                        partialEven
                    ),
                    sumEven
                );


                _mm256_store_si256(
                    reinterpret_cast<
                        __m256i*
                    >(
                        partialOdd
                    ),
                    sumOdd
                );


                raw +=
                    partialEven[0] +
                    partialEven[1] +
                    partialEven[2] +
                    partialEven[3] +
                    partialOdd[0] +
                    partialOdd[1] +
                    partialOdd[2] +
                    partialOdd[3];


                // ================================================
                // SCALAR TAIL
                //
                // Dense-1 has 1037 inputs, so AVX2 processes
                // 1032 and these final 5 are handled here.
                // ================================================

                for (
                    ;
                    inputIndex <
                    HALFKP_DENSE1_INPUT_COUNT;
                    ++inputIndex
                    ) {
                    raw +=
                        static_cast<
                            std::int64_t
                        >(
                            weights[
                                inputIndex
                            ]
                        )
                        *
                        static_cast<
                            std::int64_t
                        >(
                            combined[
                                inputIndex
                            ]
                        );
                }


                hidden1[
                    outputIndex
                ] =
                    std::clamp<
                        std::int64_t
                    >(
                        divideTowardZero(
                            raw,
                            halfKP.weightScale
                        ),
                        0,
                        halfKP.activationMax
                    );
            }


            std::array<
                std::int64_t,
                HALFKP_DENSE2_COUNT
            > hidden2{};


            for (
                int outputIndex = 0;
                outputIndex <
                HALFKP_DENSE2_COUNT;
                ++outputIndex
                ) {
                std::int64_t raw =
                    halfKP.dense2Bias[
                        outputIndex
                    ];


                const std::size_t base =
                    static_cast<
                        std::size_t
                    >(
                        outputIndex
                    )
                    *
                    HALFKP_DENSE1_COUNT;


                for (
                    int inputIndex = 0;
                    inputIndex <
                    HALFKP_DENSE1_COUNT;
                    ++inputIndex
                    ) {
                    raw +=
                        static_cast<
                            std::int64_t
                        >(
                            halfKP.dense2Weights[
                                base +
                                static_cast<
                                    std::size_t
                                >(
                                    inputIndex
                                )
                            ]
                        )
                        *
                        hidden1[
                            inputIndex
                        ];
                }


                hidden2[
                    outputIndex
                ] =
                    std::clamp<
                        std::int64_t
                    >(
                        divideTowardZero(
                            raw,
                            halfKP.weightScale
                        ),
                        0,
                        halfKP.activationMax
                    );
            }


            std::int64_t rawOutput =
                halfKP.outputBias;


            for (
                int hidden = 0;
                hidden <
                HALFKP_DENSE2_COUNT;
                ++hidden
                ) {
                rawOutput +=
                    hidden2[
                        hidden
                    ]
                    *
                    static_cast<
                        std::int64_t
                    >(
                        halfKP.outputWeights[
                            hidden
                        ]
                    );
            }


            std::int64_t centipawns =
                divideTowardZero(
                    rawOutput,
                    halfKP.outputScale
                );


            centipawns =
                std::clamp<
                    std::int64_t
                >(
                    centipawns,
                    -30000,
                    30000
                );


            return
                static_cast<int>(
                    centipawns
                );
        }

        
    } // anonymous namespace


    // ============================================================
    // HALFKP FULL REBUILD
    // ============================================================

    void rebuildHalfKPAccumulators(
        Position& pos
    ) {
        if (
            !network.loaded
            ||
            network.format !=
            NetworkFormat::HalfKP512
        ) {
            pos.halfKPValid =
                false;

            return;
        }


        pos.halfKPValid =
            false;


        if (
            !rebuildPerspective(
                pos,
                true
            )
            ||
            !rebuildPerspective(
                pos,
                false
            )
        ) {
            pos.halfKPValid =
                false;

            return;
        }


        pos.halfKPGeneration =
            networkGeneration;

        pos.halfKPValid =
            true;
    }
    int evaluateHalfKPFullRebuild(
        const Position& pos
    ) {
        if (
            !network.loaded
            ||
            network.format !=
            NetworkFormat::HalfKP512
        ) {
            return 0;
        }


        Position copy =
            pos;


        copy.halfKPValid =
            false;

        copy.halfKPGeneration =
            0;


        rebuildHalfKPAccumulators(
            copy
        );


        return
            evaluateHalfKP512(
                copy
            );
    }

    // ============================================================
    // HALFKP AFTER MAKE
    // ============================================================

    void updateHalfKPAfterMove(
        Position& pos,
        const Move& move,
        const UndoState& undo
    ) {
        if (
            !halfKPCacheUsable(
                pos
            )
        ) {
            return;
        }


        const char movingPiece =
            undo.movedPiece;


        const char destinationPiece =
            destinationPieceForMove(
                move,
                undo
            );


        const bool whiteKingMoved =
            movingPiece ==
            'K';


        const bool blackKingMoved =
            movingPiece ==
            'k';


        // --------------------------------------------------------
        // WHITE PERSPECTIVE
        // --------------------------------------------------------

        if (
            whiteKingMoved
        ) {
            if (
                !rebuildPerspective(
                    pos,
                    true
                )
            ) {
                return;
            }
        }
        else {
            applyFeatureDelta(
                pos,
                true,
                movingPiece,
                move.from,
                -1
            );


            if (
                undo.capturedPiece !=
                '.'
            ) {
                applyFeatureDelta(
                    pos,
                    true,
                    undo.capturedPiece,
                    undo.capturedSquare,
                    -1
                );
            }


            applyFeatureDelta(
                pos,
                true,
                destinationPiece,
                move.to,
                +1
            );


            applyCastlingRookForward(
                pos,
                true,
                move,
                movingPiece
            );
        }


        // --------------------------------------------------------
        // BLACK PERSPECTIVE
        // --------------------------------------------------------

        if (
            blackKingMoved
        ) {
            if (
                !rebuildPerspective(
                    pos,
                    false
                )
            ) {
                return;
            }
        }
        else {
            applyFeatureDelta(
                pos,
                false,
                movingPiece,
                move.from,
                -1
            );


            if (
                undo.capturedPiece !=
                '.'
            ) {
                applyFeatureDelta(
                    pos,
                    false,
                    undo.capturedPiece,
                    undo.capturedSquare,
                    -1
                );
            }


            applyFeatureDelta(
                pos,
                false,
                destinationPiece,
                move.to,
                +1
            );


            applyCastlingRookForward(
                pos,
                false,
                move,
                movingPiece
            );
        }


        if (
            !pos.halfKPValid
        ) {
            return;
        }


        pos.halfKPGeneration =
            networkGeneration;
    }


    // ============================================================
    // HALFKP AFTER UNDO
    // ============================================================

    void updateHalfKPAfterUndo(
        Position& pos,
        const Move& move,
        const UndoState& undo
    ) {
        if (
            !halfKPCacheUsable(
                pos
            )
        ) {
            return;
        }


        const char movingPiece =
            undo.movedPiece;


        const char destinationPiece =
            destinationPieceForMove(
                move,
                undo
            );


        const bool whiteKingMoved =
            movingPiece ==
            'K';


        const bool blackKingMoved =
            movingPiece ==
            'k';


        // --------------------------------------------------------
        // WHITE PERSPECTIVE
        // --------------------------------------------------------

        if (
            whiteKingMoved
        ) {
            if (
                !rebuildPerspective(
                    pos,
                    true
                )
            ) {
                return;
            }
        }
        else {
            applyCastlingRookBackward(
                pos,
                true,
                move,
                movingPiece
            );


            applyFeatureDelta(
                pos,
                true,
                destinationPiece,
                move.to,
                -1
            );


            if (
                undo.capturedPiece !=
                '.'
            ) {
                applyFeatureDelta(
                    pos,
                    true,
                    undo.capturedPiece,
                    undo.capturedSquare,
                    +1
                );
            }


            applyFeatureDelta(
                pos,
                true,
                movingPiece,
                move.from,
                +1
            );
        }


        // --------------------------------------------------------
        // BLACK PERSPECTIVE
        // --------------------------------------------------------

        if (
            blackKingMoved
        ) {
            if (
                !rebuildPerspective(
                    pos,
                    false
                )
            ) {
                return;
            }
        }
        else {
            applyCastlingRookBackward(
                pos,
                false,
                move,
                movingPiece
            );


            applyFeatureDelta(
                pos,
                false,
                destinationPiece,
                move.to,
                -1
            );


            if (
                undo.capturedPiece !=
                '.'
            ) {
                applyFeatureDelta(
                    pos,
                    false,
                    undo.capturedPiece,
                    undo.capturedSquare,
                    +1
                );
            }


            applyFeatureDelta(
                pos,
                false,
                movingPiece,
                move.from,
                +1
            );
        }


        if (
            !pos.halfKPValid
        ) {
            return;
        }


        pos.halfKPGeneration =
            networkGeneration;
    }

    // ============================================================
    // V1 FEATURE INDEX
    // ============================================================

    int nnueFeatureIndex(
        char piece,
        int square
    ) {
        if (
            square < 0 ||
            square >= 64
        ) {
            return -1;
        }


        const int pieceIndex =
            pieceIndexForChar(
                piece
            );


        if (
            pieceIndex < 0
        ) {
            return -1;
        }


        return
            pieceIndex *
            64 +
            square;
    }


    // ============================================================
    // HALFKP FEATURE INDEX
    // ============================================================

    int halfKPFeatureIndex(
        bool perspectiveWhite,
        int kingSquare,
        char piece,
        int pieceSquare
    ) {
        if (
            kingSquare < 0 ||
            kingSquare >= 64 ||
            pieceSquare < 0 ||
            pieceSquare >= 64
        ) {
            return -1;
        }


        if (
            piece == '.' ||
            pieceIsKing(
                piece
            )
        ) {
            return -1;
        }


        const int pieceType =
            halfKPPieceTypeIndex(
                piece
            );


        if (
            pieceType < 0
        ) {
            return -1;
        }


        const bool ownPiece =
            pieceIsWhite(
                piece
            )
            ==
            perspectiveWhite;


        const int pieceClass =
            ownPiece
            ?
            pieceType
            :
            5 +
            pieceType;


        const int orientedKing =
            orientSquare(
                perspectiveWhite,
                kingSquare
            );


        const int orientedPiece =
            orientSquare(
                perspectiveWhite,
                pieceSquare
            );


        const int feature =
            orientedKing *
            (
                10 *
                64
            )
            +
            pieceClass *
            64
            +
            orientedPiece;


        if (
            feature < 0 ||
            feature >=
            HALFKP_FEATURE_COUNT
        ) {
            return -1;
        }


        return feature;
    }


    // ============================================================
    // LOAD NETWORK
    // ============================================================

    bool loadNNUE(
        const std::string& path,
        std::string* errorMessage
    ) {
        auto fail =
            [
                errorMessage
            ](
                const std::string& message
                ) {
                if (
                    errorMessage !=
                    nullptr
                ) {
                    *errorMessage =
                        message;
                }


                return false;
            };


        std::ifstream input(
            path,
            std::ios::binary
        );


        if (
            !input
        ) {
            return
                fail(
                    "Unable to open NNUE file: " +
                    path
                );
        }


        std::array<char, 8>
            magic{};


        input.read(
            magic.data(),
            static_cast<
                std::streamsize
            >(
                magic.size()
            )
        );


        if (
            !input
        ) {
            return
                fail(
                    "NNUE file ended while reading magic/header."
                );
        }


        Network candidate;


        bool result =
            false;


        if (
            magic ==
            NNUE_V1_MAGIC
        ) {
            result =
                loadV1(
                    input,
                    path,
                    candidate,
                    errorMessage
                );
        }
        else if (
            magic ==
            NNUE_V2_MAGIC
        ) {
            result =
                loadHalfKP512(
                    input,
                    path,
                    candidate,
                    errorMessage
                );
        }
        else {
            return
                fail(
                    "Invalid KnightBot NNUE magic/header."
                );
        }


        if (
            !result
        ) {
            return false;
        }


        network =
            std::move(
                candidate
            );
        ++networkGeneration;

        if (
            networkGeneration == 0
        ) {
            networkGeneration = 1;
        }

        if (
            errorMessage !=
            nullptr
        ) {
            errorMessage->clear();
        }


        return true;
    }


    // ============================================================
    // UNLOAD
    // ============================================================

    void unloadNNUE() {
        network =
            Network{};


        ++networkGeneration;

        if (
            networkGeneration == 0
        ) {
            networkGeneration = 1;
        }
    }


    bool nnueLoaded() {
        return
            network.loaded;
    }


    std::string nnueLoadedFile() {
        return
            network.sourceFile;
    }


    std::string nnueFormatName() {
        switch (
            network.format
            ) {
        case NetworkFormat::V1:
            return
                "NNUE v1";

        case NetworkFormat::HalfKP512:
            return
                "HalfKP-512";

        case NetworkFormat::None:
        default:
            return
                "none";
        }
    }


    int nnueFormatVersion() {
        return
            static_cast<int>(
                network.format
            );
    }


    // ============================================================
    // ENABLE
    // ============================================================

    void setNNUEEnabled(
        bool enabled
    ) {
        useNNUE =
            enabled;
    }


    bool nnueEnabled() {
        return
            useNNUE;
    }


    // ============================================================
    // INFERENCE DISPATCH
    // ============================================================

    int evaluateNNUE(
        const Position& pos
    ) {
        if (
            !network.loaded
        ) {
            return 0;
        }


        switch (
            network.format
            ) {
        case NetworkFormat::V1:
            return
                evaluateV1(
                    pos
                );


        case NetworkFormat::HalfKP512:
            return
                evaluateHalfKP512(
                    pos
                );


        case NetworkFormat::None:
        default:
            return 0;
        }
    }


    // ============================================================
    // V1 COMPATIBILITY TEST NETWORK
    // ============================================================

    bool writeNNUECompatibilityTestNetwork(
        const std::string& path,
        std::string* errorMessage
    ) {
        auto fail =
            [
                errorMessage
            ](
                const std::string& message
                ) {
                if (
                    errorMessage !=
                    nullptr
                ) {
                    *errorMessage =
                        message;
                }


                return false;
            };


        std::ofstream output(
            path,
            std::ios::binary |
            std::ios::trunc
        );


        if (
            !output
        ) {
            return
                fail(
                    "Unable to create test NNUE file: " +
                    path
                );
        }


        output.write(
            NNUE_V1_MAGIC.data(),
            static_cast<
                std::streamsize
            >(
                NNUE_V1_MAGIC.size()
            )
        );


        const std::uint32_t version =
            NNUE_V1_FORMAT_VERSION;

        const std::uint32_t inputCount =
            NNUE_V1_INPUT_COUNT;

        const std::uint32_t hiddenCount =
            NNUE_V1_HIDDEN_COUNT;


        const std::int32_t activationMax =
            10000;


        const std::int32_t outputScale =
            1;


        if (
            !writeValue(
                output,
                version
            )
            ||
            !writeValue(
                output,
                inputCount
            )
            ||
            !writeValue(
                output,
                hiddenCount
            )
            ||
            !writeValue(
                output,
                activationMax
            )
            ||
            !writeValue(
                output,
                outputScale
            )
        ) {
            return
                fail(
                    "Unable to write NNUE V1 header."
                );
        }


        // ========================================================
        // HIDDEN BIASES
        // ========================================================

        std::array<
            std::int32_t,
            NNUE_V1_HIDDEN_COUNT
        > hiddenBias{};


        output.write(
            reinterpret_cast<
                const char*
            >(
                hiddenBias.data()
            ),
            static_cast<
                std::streamsize
            >(
                sizeof(std::int32_t) *
                NNUE_V1_HIDDEN_COUNT
            )
        );


        // ========================================================
        // INPUT WEIGHTS
        // ========================================================
        //
        // hidden 0 = White material
        // hidden 1 = Black material
        // ========================================================

        const std::size_t count =
            static_cast<
                std::size_t
            >(
                NNUE_V1_INPUT_COUNT
            )
            *
            NNUE_V1_HIDDEN_COUNT;


        std::vector<std::int16_t>
            inputWeights(
                count,
                0
            );


        for (
            int pieceIndex = 0;
            pieceIndex <
            PIECE_COUNT;
            ++pieceIndex
            ) {
            const int value =
                compatibilityMaterialValue(
                    pieceIndex
                );


            const int hidden =
                pieceIndex <=
                WK
                ? 0
                : 1;


            for (
                int square = 0;
                square < 64;
                ++square
                ) {
                const int feature =
                    pieceIndex *
                    64 +
                    square;


                const std::size_t index =
                    static_cast<
                        std::size_t
                    >(
                        feature
                    )
                    *
                    NNUE_V1_HIDDEN_COUNT
                    +
                    hidden;


                inputWeights[
                    index
                ] =
                    static_cast<
                        std::int16_t
                    >(
                        value
                    );
            }
        }


        output.write(
            reinterpret_cast<
                const char*
            >(
                inputWeights.data()
            ),
            static_cast<
                std::streamsize
            >(
                sizeof(std::int16_t) *
                inputWeights.size()
            )
        );


        // ========================================================
        // OUTPUT WEIGHTS
        // ========================================================

        std::array<
            std::int16_t,
            NNUE_V1_HIDDEN_COUNT
        > outputWeights{};


        outputWeights[0] =
            1;

        outputWeights[1] =
            -1;


        output.write(
            reinterpret_cast<
                const char*
            >(
                outputWeights.data()
            ),
            static_cast<
                std::streamsize
            >(
                sizeof(std::int16_t) *
                NNUE_V1_HIDDEN_COUNT
            )
        );


        const std::int32_t outputBias =
            0;


        if (
            !writeValue(
                output,
                outputBias
            )
        ) {
            return
                fail(
                    "Unable to finish NNUE V1 file."
                );
        }


        if (
            !output
        ) {
            return
                fail(
                    "Error writing compatibility NNUE V1 file."
                );
        }


        if (
            errorMessage !=
            nullptr
        ) {
            errorMessage->clear();
        }


        return true;
    }

} // namespace chess
