#pragma once

#include "chess.hpp"

namespace chess {

    // ============================================================
    // EVALUATION BREAKDOWN
    // ============================================================
    //
    // All scores are stored in centipawns.
    //
    // Positive = White better
    // Negative = Black better
    //
    // MG = middlegame score
    // EG = endgame score
    //
    // ============================================================

    struct EvaluationBreakdown {
        int materialAndPstMg = 0;
        int materialAndPstEg = 0;

        int bishopPairMg = 0;
        int bishopPairEg = 0;

        int pawnStructureMg = 0;
        int pawnStructureEg = 0;

        int pieceActivityMg = 0;
        int pieceActivityEg = 0;

        int kingSafetyMg = 0;
        int kingSafetyEg = 0;

        int tempoMg = 0;
        int tempoEg = 0;

        int phase = 0;

        int finalScore = 0;

        bool insufficientMaterial = false;
    };


    // ============================================================
    // PIECE VALUE
    // ============================================================
    //
    // Absolute material value used by search heuristics such as
    // SEE and move ordering.
    //
    // These are centipawn-like values.
    //
    // ============================================================

    int pieceValue(
        char piece
    );


    // ============================================================
    // FULL STATIC EVALUATION
    // ============================================================
    //
    // Positive = White better
    // Negative = Black better
    //
    // ============================================================

    int evaluate(
        const Position& pos
    );


    // ============================================================
    // ACTIVE EVALUATOR
    // ============================================================
    //
    // Uses NNUE when it is enabled and a valid network is loaded.
    // Otherwise falls back to the handcrafted evaluator.
    // Positive = White better.
    // ============================================================

    int evaluateActive(
        const Position& pos
    );


    // ============================================================
    // DETAILED STATIC EVALUATION
    // ============================================================
    //
    // Returns the individual components used to produce the
    // final static evaluation.
    //
    // Useful for debugging and tuning the evaluator.
    //
    // ============================================================

    EvaluationBreakdown evaluateDetailed(
        const Position& pos
    );


    // ============================================================
    // SIDE-TO-MOVE EVALUATION
    // ============================================================
    //
    // Negamax-friendly evaluation.
    //
    // Positive = side to move better
    // Negative = side to move worse
    //
    // ============================================================

    int evaluateForSideToMove(
        const Position& pos
    );

} // namespace chess
