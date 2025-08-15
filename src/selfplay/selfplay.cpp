#include <fstream>
#include <string>
#include <filesystem>
#include <chrono>
#include <thread>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "include/selfplay/selfplay.hpp"
#include "include/utils/random.hpp"
#include "include/utils/functions.hpp"



// Helper function to format seconds into HH:MM:SS format
std::string formatTime(int total_seconds) {
    int hours = total_seconds / 3600;
    int minutes = (total_seconds % 3600) / 60;
    int seconds = total_seconds % 60;
    
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(2) << hours << ":"
        << std::setfill('0') << std::setw(2) << minutes << ":"
        << std::setfill('0') << std::setw(2) << seconds;
    return oss.str();
}

SelfPlay::SelfPlay(int total_games, int sims_per_move, unsigned int threads, float resign_threshold, int nn_cache_size, bool trust_val, torch::jit::script::Module& nnet, torch::Device device, size_t ttable_size, float temperature_start) :
                    total_games(total_games), sims_per_move(sims_per_move), threads(threads), resign_threshold(resign_threshold), nn_cache_size(nn_cache_size), trust_val(trust_val),
                    nnet(nnet), device(device), ttable_size(ttable_size), temperature_start(temperature_start) {transposition_table.set_size(ttable_size);}

void SelfPlay::run() {
    ThreadPool pool(threads);
    for (unsigned int i = 0; i < total_games; ++i) {
        pool.enqueueTask([this] { this->selfPlayGame(); });
    }

    // every 5 seconds, print the game info
    auto start_time = std::chrono::steady_clock::now();
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        
        clearTerminal();
        std::cout << "=== Self-Play Progress ===\n";
        std::cout << "Total Games: " << total_games << "\n";
        std::cout << "Completed Games: " << game_index - threads << "\n";
        
        if (!game_info.empty()) {
            std::cout << "=== Active Games ===\n";
            for (const auto& [thread_id, info] : game_info) {
                std::cout << "Thread " << thread_id << ": " << info << "\n";
            }
        }
        
        std::cout << "\n=== Performance Info ===\n";
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
        std::cout << "Elapsed Time: " << formatTime(elapsed.count()) << "\n";
        auto moves_per_second = static_cast<float>(num_moves.load(std::memory_order_relaxed)) / elapsed.count();
        std::cout << "Moves per Second: " << std::fixed << std::setprecision(2) << moves_per_second << "\n";
        
        // Check if all games are completed
        if (game_index >= total_games) {
            std::cout << "\nAll games completed! SelfPlay finished.\n";
            break;
        }
    }
}

void SelfPlay::selfPlayGame() {
    auto index = getGameIndex();
    auto temperature = temperature_start;
    chess::Board startState = chess::Board(startFen);
    std::vector<chess::Board> traversed = {};
    std::string pgn_moves = "";
    std::vector<float> policyBuffer;
    std::vector<float> valueBuffer;
    auto result = 0;

    // decide whether to honor the resign threshold
    auto res_threshold = 1.0f;
    float probability = 0.95;
    if (randomBernoulli(probability)) { res_threshold = resign_threshold; }

    std::string directoryPath = "selfplay_games/game-" + std::to_string(index);
    std::filesystem::path dir {directoryPath};
    std::filesystem::create_directories(dir);

    std::ofstream PolicyLabels(directoryPath + "/policy.bin", std::ios::binary | std::ios::app);
    std::ofstream ValueLabels(directoryPath + "/q_values.bin", std::ios::binary | std::ios::app);

    uint8_t progress = 0;
    std::string thread_id = std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    game_info[thread_id] = "Game #" + std::to_string(index) + ", Move: " + std::to_string(1) + ", RT: " + std::to_string(res_threshold);
    
    for (int turns = 0; turns < 256; ++turns) {
        game_info[thread_id] = "Game #" + std::to_string(index) + ", Move: " + std::to_string(turns+1) + ", RT: " + std::to_string(res_threshold);
        Container container;
        auto rootNode = new Node(container, startState, progress);
        auto newSearch = Search(rootNode, container, traversed, transposition_table, nnet, device, 
                            sims_per_move, 1, nn_cache_size, false);
        if (turns == 30) {temperature = temperature_end;}
        newSearch.startSearch(true);
        auto move_map = get_move_map(rootNode, trust_val);
        valueBuffer.emplace_back(newSearch.getRootQ());
        // // for debugging
        // for (const auto& move : move_map) {
        //     std::cout << move.first << ", " << move.second << '\n';
        // }
        auto policy = policy_map::get_move_to_policy(move_map, startState.sideToMove());
        for (size_t j = 0; j < 4672; ++j) {
            policyBuffer.push_back(policy[j]);
        }
        auto move = newSearch.selectMove(false, temperature, res_threshold);
        num_moves.fetch_add(1, std::memory_order_relaxed);
        progress = newSearch.rootNode->moves_since_cpm;
        
        if (move.second != -1) {
            if (move.second != 1 && startState.sideToMove() == chess::Color::BLACK) {
                result = -(move.second - 1);
            }
            else if (move.second != 1 && startState.sideToMove() == chess::Color::WHITE) {
                result = (move.second - 1);
            }

            if (turns%2 == 0) {
                pgn_moves += std::to_string((turns/2)+1) + ". ";
            }
            // Add the move to the PGN string
            pgn_moves += chess::uci::moveToSan(startState, move.first) + " ";
            startState.makeMove(move.first);
            break;
        }
        
        if (turns%2 == 0) {
            pgn_moves += std::to_string((turns/2)+1) + ". ";
        }
        // Add the move to the PGN string
        pgn_moves += chess::uci::moveToSan(startState, move.first) + " ";
        startState.makeMove(move.first);
    }

    if (!policyBuffer.empty()) {
        PolicyLabels.write(reinterpret_cast<const char*>(policyBuffer.data()), policyBuffer.size() * sizeof(float));
    }
    PolicyLabels.close();

    if (!valueBuffer.empty()) {
        ValueLabels.write(reinterpret_cast<const char*>(valueBuffer.data()), valueBuffer.size() * sizeof(float));
    }
    ValueLabels.close();

    std::string result_string = "1/2-1/2";
    if (result == 1) {
        result_string = "1-0";
    }
    else if (result == -1) {
        result_string = "0-1";
    }

    // Once the game is over, write the PGN data to a file
    std::ofstream pgn_file(directoryPath + "/game.pgn");

    pgn_file << "[Event \"Training Game\"]\n";
    pgn_file << "[Site \"?\"]\n";
    pgn_file << "[Date \"????.??.??\"]\n";
    pgn_file << "[Round \"Start Temp of " + std::to_string(temperature_start) + "\"]\n";
    pgn_file << "[White \"NarChesser\"]\n";
    pgn_file << "[Black \"NarChesser\"]\n";
    pgn_file << "[Result \"" + result_string + "\"]\n\n";
    // Write the moves
    pgn_file << pgn_moves << result_string;
    pgn_file.close();
}

std::unordered_map<chess::Move, float> SelfPlay::get_move_map(const Node* root, bool trust_val) {
    std::unordered_map<chess::Move, float> move_map;
    if (trust_val) {
        for (const auto& child : root->children) {
            move_map.insert({child->move, static_cast<float>(child->visits)/static_cast<float>(sims_per_move)});
        }
        return move_map;
    }
    else {
        for (const auto& child : root->children) {
            move_map.insert({child->move, child->policy});
        }
        return move_map;
    }
}
