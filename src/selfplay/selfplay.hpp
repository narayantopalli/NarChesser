#pragma once
#include "../search/search.hpp"

struct SelfPlay {
    std::string startFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    int total_games;
    int sims_per_move;
    int threads;
    float resign_threshold;
    float temperature_start;
    int nn_cache_size;
    int game_index = 0;
    bool trust_val;
    std::map<std::string, std::string> game_info;
    std::map<std::string, std::string> game_info_old;
    std::atomic<int> num_moves;

    size_t ttable_size;

    torch::jit::script::Module& nnet;
    torch::Device device;

    std::mutex indexMutex;
    TranspositionTable<uint64_t, std::pair<std::unordered_map<chess::Move, float>, float>> transposition_table;
    
    SelfPlay(int total_games, int sims_per_move, unsigned int threads, float resign_threshold, int nn_cache_size, bool trust_val, torch::jit::script::Module& nnet, torch::Device device, size_t ttable_size, float temperature_start);
    void selfPlayGame();
    void run();
    inline int getGameIndex();
    std::unordered_map<chess::Move, float> get_move_map(const Node* root, bool trust_val = true);
};

inline int SelfPlay::getGameIndex() {
    std::lock_guard<std::mutex> guard(indexMutex);
    auto index = game_index;
    ++game_index;
    return index;
}
