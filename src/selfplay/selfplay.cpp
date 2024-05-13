#include <fstream>
#include <string>
#include <filesystem>
#include "selfplay.hpp"

SelfPlay::SelfPlay(int total_games, int sims_per_move, unsigned int threads, float resign_threshold, int nn_cache_size, bool trust_val, torch::jit::script::Module& nnet, torch::Device device, size_t ttable_size, float temperature_start, bool tactic_bonus) :
                    total_games(total_games), sims_per_move(sims_per_move), threads(threads), resign_threshold(resign_threshold), nn_cache_size(nn_cache_size), trust_val(trust_val),
                    nnet(nnet), device(device), ttable_size(ttable_size), temperature_start(temperature_start), tactic_bonus(tactic_bonus) {transposition_table.set_size(ttable_size);}

void SelfPlay::run() {
    ThreadPool pool(threads);
    for (unsigned int i = 0; i < total_games; ++i) {
        pool.enqueueTask([this] { this->selfPlayGame(); });
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
    std::random_device rd;
    std::mt19937 gen(rd());
    float probability = 0.95;
    std::bernoulli_distribution d(probability);
    if (d(gen)) { res_threshold = resign_threshold; }

    std::string directoryPath = "selfplay_games/game-" + std::to_string(index);
    std::filesystem::path dir {directoryPath};
    std::filesystem::create_directories(dir);

    std::ofstream PolicyLabels(directoryPath + "/policy.bin", std::ios::binary | std::ios::app);
    std::ofstream ValueLabels(directoryPath + "/q_values.bin", std::ios::binary | std::ios::app);

    for (int turns = 0; turns < 256; ++turns) {
        if (turns%19 == 0) {
            std::cout << "game-" + std::to_string(index) + ", resign_threshold: "+ std::to_string(res_threshold) + ", move: " + std::to_string(turns+1) + '\n';
        }
        else {
            std::cout << ".\n";
        }
        Container container;
        auto rootNode = new Node(container, startState);
        auto newSearch = Search(rootNode, container, traversed, transposition_table, nnet, device, 
                            sims_per_move, 1, nn_cache_size, false, tactic_bonus);
        if (turns == 30) {temperature = temperature_end;}
        newSearch.startSearch(true);
        auto move_map = get_move_map(rootNode, trust_val);
        valueBuffer.emplace_back(newSearch.getQ());
        // // for debugging
        // for (const auto& move : move_map) {
        //     std::cout << move.first << ", " << move.second << '\n';
        // }
        auto policy = policy_map::get_move_to_policy(move_map, startState.sideToMove());
        for (size_t j = 0; j < 4672; ++j) {
            policyBuffer.push_back(policy[j]);
        }
        auto move = newSearch.selectMove(false, temperature, res_threshold);
        
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

    pgn_file << "[Event \"Self-play game\"]\n";
    pgn_file << "[Site \"?\"]\n";
    pgn_file << "[Date \"????.??.??\"]\n";
    pgn_file << "[Round \"Start Temp of " + std::to_string(temperature_start) + "\"]\n";
    pgn_file << "[White \"Self-play AI\"]\n";
    pgn_file << "[Black \"Self-play AI\"]\n";
    pgn_file << "[Result \"" + result_string + "\"]\n\n";
    // Write the moves
    pgn_file << pgn_moves << result_string;
    pgn_file.close();
    std::cout << result_string + ", game-" + std::to_string(index) + '\n';
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
