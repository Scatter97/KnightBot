#pragma once

#include "chess.hpp"

namespace chess {

	// Absolute material value used by search heuristics such as SEE
	// and move ordering.
	//
	// These are centipawn-like values.
	int pieceValue(char piece);


	// Full static evaluation.
	//
	// Positive = White better
	// Negative = Black better
	int evaluate(const Position& pos);


	// Negamax-friendly evaluation.
	//
	// Positive = side to move better
	// Negative = side to move worse
	int evaluateForSideToMove(const Position& pos);

} // namespace chess