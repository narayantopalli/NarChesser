#pragma once

#include <torch/script.h>
#include "../planes.hpp"
#include "../search/node.hpp"

struct EncodedState {
    std::unique_ptr<Bitboard[]> encodedState;
    EncodedState(const Node* node, const std::vector<chess::Board>& traversed, const uint8_t history);
    inline torch::Tensor toTensor();
    const uint8_t history;
    uint8_t totalPlanes;
};

// Function to convert a bitboard to a 2D tensor representation
inline torch::Tensor EncodedState::toTensor() {
    auto tensor = torch::zeros({totalPlanes, 8, 8}, torch::kHalf);
    for (int board = 0; board < totalPlanes; ++board) {
        for (int i = 0; i < 64; ++i) {
            // Check if the ith bit is set
            if (encodedState[board] & (1ULL << i)) {
                // Calculate row and column from bit position
                int row = i / 8;
                int col = i % 8;
                
                // Set the corresponding tensor element
                tensor[board][row][col] = 1.0;
            }
        }
    }
    
    return tensor;
}
