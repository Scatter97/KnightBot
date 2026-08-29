#include "training.hpp"

#include <fstream>
#include <string>

namespace chess {

    bool appendTrainingPosition(
        const std::string& path,
        const Position& pos,
        int targetCentipawns,
        std::string* errorMessage
    ) {
        std::ofstream output(
            path,
            std::ios::out |
            std::ios::app
        );


        if (
            !output
            ) {
            if (
                errorMessage !=
                nullptr
                ) {
                *errorMessage =
                    "Unable to open training-data file: " +
                    path;
            }


            return false;
        }


        output
            <<
            toFEN(
                pos
            )
            <<
            '\t'
            <<
            targetCentipawns
            <<
            '\n';


        if (
            !output
            ) {
            if (
                errorMessage !=
                nullptr
                ) {
                *errorMessage =
                    "Unable to write training position.";
            }


            return false;
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
