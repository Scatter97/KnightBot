#pragma once

#include "chess.hpp"

namespace chess {

	// Returns the basic value of a piece in centipawns.
	//
	// Pawn   = 100
	// Knight = 320
	// Bishop = 330
	// Rook   = 500
	// Queen  = 900
	int pieceValue(char piece);


	// Evaluates from White's perspective.
	//
	// Positive = White is better
	// Negative = Black is better
	int evaluate(const Position& pos);


	// Evaluates from the perspective of the side to move.
	//
	// Positive = side to move is better
	// Negative = side to move is worse
	int evaluateForSideToMove(const Position& pos);

} // namespace chess