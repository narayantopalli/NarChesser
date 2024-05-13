#pragma once
#include "../src/chess.hpp"
#include <memory>
#include <array>

constexpr int BOARD_SIZE = 8;
constexpr int PLANES = 73;
constexpr int PROMOTIONS[] = {2, 3, 4};
// 2 is Bishop, 3 is Rook, 4 is Queen, default is Knight

constexpr int promotion_to_index(chess::PieceType promotion) {
    if (promotion == chess::PieceType::BISHOP) {
        return 2;
    }
    else if (promotion == chess::PieceType::ROOK) {
        return 3;
    }
    else if (promotion == chess::PieceType::QUEEN) {
        return 4;
    }
    else {
        return 1;
    }
}

class Directions {
public:
    enum Type { N, NE, E, SE, S, SW, W, NW };

    static constexpr std::pair<int, int> queenMoves[] = {
        {0, 1}, {1, 1}, {1, 0}, {1, -1},
        {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}
    };

    static constexpr std::pair<int, int> knightMoves[] = {
        {1, 2}, {2, 1}, {2, -1}, {1, -2},
        {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}
    };
};

using PolicyMap = std::array<int, PLANES * BOARD_SIZE * BOARD_SIZE>;
constexpr PolicyMap initializePolicyMap() {
    PolicyMap policyMap{};
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            int from[] = {rank, file};
            int plane = 0;
            for (const auto& QueenMove : Directions::queenMoves) {
                for (int step = 1; step < 8; ++step) {
                    int newRank = rank + step*QueenMove.second;
                    int newFile = file + step*QueenMove.first;
                    if ((0 <= newRank && newRank < 8) && (0 <= newFile && newFile < 8)) {
                        policyMap[(rank*BOARD_SIZE*PLANES) + (file*PLANES) + plane] = (8*newRank) + newFile;
                    }
                    else {
                        policyMap[(rank*BOARD_SIZE*PLANES) + (file*PLANES) + plane] = -1;
                    }
                    ++plane;
                }
            }
            for (const auto& KnightMove : Directions::knightMoves) {
                int newRank = rank + KnightMove.second;
                int newFile = file + KnightMove.first;
                if ((0 <= newRank && newRank < 8) && (0 <= newFile && newFile < 8)) {
                    policyMap[(rank*BOARD_SIZE*PLANES) + (file*PLANES) + plane] = (8*newRank) + newFile;
                }
                else {
                    policyMap[(rank*BOARD_SIZE*PLANES) + (file*PLANES) + plane] = -1;
                }
                ++plane;
            }
            for (const auto& promotion : PROMOTIONS) {
                for (const auto& direction : {0, 1, 7}) {
                    int newRank = rank + (Directions::queenMoves)[direction].second;
                    int newFile = file + (Directions::queenMoves)[direction].first;
                    if ((newRank == 7) && (0 <= newFile && newFile < 8)) {
                        policyMap[(rank*BOARD_SIZE*PLANES) + (file*PLANES) + plane] = ((8*newRank) + newFile)*promotion;
                    }
                    else {
                        policyMap[(rank*BOARD_SIZE*PLANES) + (file*PLANES) + plane] = -1;
                    }
                    ++plane;
                }
            }
        }
    }

    // Here, we would initialize the policy map with static values if applicable.
    // Dynamic probability assignments based on game state cannot be done at compile time.
    return policyMap;
}

constexpr PolicyMap policyMap = initializePolicyMap();

const int totalSize = PLANES*BOARD_SIZE*BOARD_SIZE;
std::bitset<totalSize> get_move_to_policy(chess::Move move, chess::Color color) {

    std::bitset<totalSize> policyIndex;
    int c = static_cast<int>(color);
    int from_rank_index = (c == 0 ? static_cast<int>(move.from().rank()) : 7 - static_cast<int>(move.from().rank()));
    int from_file_index = static_cast<int>(move.from().file());
    int dest_rank_index = (c == 0 ? static_cast<int>(move.to().rank()) : 7 - static_cast<int>(move.to().rank()));
    int dest_file_index = static_cast<int>(move.to().file());
    int promotion_to_int = promotion_to_index(move.promotionType());
    for (int plane = 0; plane < PLANES; ++plane) {
        bool rightSquare = (policyMap[(from_rank_index*BOARD_SIZE*PLANES) + (from_file_index*PLANES) + plane] == ((BOARD_SIZE*dest_rank_index) + dest_file_index)*promotion_to_int);
        if (rightSquare) {
            int index = (from_rank_index * BOARD_SIZE * PLANES) + (from_file_index * PLANES) + plane;
            policyIndex.set(index);
            break;
        }
    }

    return policyIndex;

}
