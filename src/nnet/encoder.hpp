#pragma once

#include <torch/script.h> // Include the PyTorch library
#include "../planes.hpp" // Relative path to the custom planes header
#include "../search/node.hpp" // Relative path to the Node structure for search algorithm

struct EncodedState {
    std::unique_ptr<Bitboard[]> encodedState; // Dynamically allocated array of Bitboards
    EncodedState(const Node* node, const std::vector<chess::Board>& traversed, const uint8_t history);
    inline torch::Tensor toTensor();
    const uint8_t history;
    uint8_t totalPlanes;
};

inline torch::Tensor EncodedState::toTensor() {
    // Create a 8x8 tensor initialized with zeros
    auto tensor = torch::zeros({totalPlanes, 8, 8}, torch::kHalf);
    // Iterate over all bitboards in encoded state
    for (int board = 0; board < totalPlanes; ++board) {
        // Iterate over all bits in the bitboard
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
