#pragma once

#include "chess.hpp"

using namespace chess;

constexpr Bitboard Bitboard::reverseRanks() {
    Bitboard b = *this;
    Bitboard reversed = 0;
    for (int rank = 0; rank < 8; ++rank) {
        // Extract the bits of the current rank
        Bitboard rankBits = (b >> (rank * 8)) & 0xFF;
        // Shift them to their new position
        reversed |= rankBits << ((7 - rank) * 8);
    }
    return reversed;
}

constexpr int colorToInt(Color color) {
    return static_cast<int>(color);
}

constexpr int get_def_result(Color c, GameResult result) {
    if (result == GameResult::LOSE) {
        if (c == Color::WHITE) {
            return -1;
        }
        else {
            return 1;
        }
    }
    else {
        return 0;
    }
}

constexpr int color_to_result(Color c, int result) {
    if (result == 0) {
        return 0;
    }
    else {
        int color_to_int = static_cast<int>(c);
        return result*(1-(2*color_to_int));
    }
}