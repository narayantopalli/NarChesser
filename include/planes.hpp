#pragma once

#include "include/chess.hpp"
#include "include/manipulators.hpp"
#include <vector>
#include <array>
#include <memory> 

namespace planes {
    extern std::unique_ptr<Bitboard[]> toPlane(const Board& board, Color color);
    extern std::unique_ptr<Bitboard[]> extraPlanes(const Board& board);
    extern std::unique_ptr<Bitboard[]> emptyPlane();
}
