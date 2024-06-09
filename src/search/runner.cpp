// #include <crtdbg.h>

#include "search.hpp"
#include "..\selfplay\selfplay.hpp"
#include "..\config.hpp"
#include <chrono>
#include <torch/script.h>
#include <torch/torch.h>
#include <fstream>
#include <filesystem>

std::array<std::string, 8> test_positions = 
                {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 0",
                "1rb4r/pkPp3p/1b1P3n/1Q6/N3Pp2/8/P1P3PP/7K w - - 1 1",
                "rnbqkbn1/ppppp3/7r/6pp/3P1p2/3BP1B1/PPP2PPP/RN1QK1NR w - - 1 1",
                "r1b3nr/ppqk1Bbp/2pp4/4P1B1/3n4/3P4/PPP2QPP/R4RK1 w - - 1 0",
                "r1b1kb1r/pppp1ppp/5q2/4n3/3KP3/2N3PN/PPP4P/R1BQ1B1R b kq - 0 1",
                "r1bq1b1r/1p3ppp/p4N2/1B1PpkB1/2Q5/8/PPP2PPP/R3K2R w KQ - 2 13",
                "3r1bk1/1b2qp1p/2p1p1p1/1pP1P2P/1Pp1B2Q/2B3P1/5PK1/2R5 w - - 3 30"};

void selfPlay(torch::jit::script::Module& nnet, torch::Device device) {

    unsigned int total_games, num_simulations, nn_cache_size, tactic_bonus;

    std::cout << "Enter the total number of self-play games: ";
    std::cin >> total_games;

    std::cout << "Enter the number of simulations (higher gives more accuracy but longer training time): ";
    std::cin >> num_simulations;

    std::cout << "Enter the eval cache size (higher values will give more breadth but less depth to search and potentially faster training): ";
    std::cin >> nn_cache_size;

    std::cout << "Train tactics (not fully reliable): ";
    std::cin >> tactic_bonus;
    bool tb = static_cast<bool>(tactic_bonus);

    std::cout << "Total games: " << total_games << std::endl;
    std::cout << "Number of simulations: " << num_simulations << std::endl;
    std::cout << "Eval cache size: " << nn_cache_size << std::endl;
    std::cout << "Number of threads: " << thread_count << std::endl;
    std::cout << "Win threshold: " << resign_eval_threshold << std::endl;
    std::cout << "Starting temperature: " << temperature_start << std::endl;
    std::cout << "Train tactics: " << tb << std::endl;

    auto self_play = SelfPlay(total_games, num_simulations, thread_count, resign_eval_threshold, nn_cache_size, true, nnet, device, transposition_table_size, temperature_start, tb);
    std::cout << "transposition table size: " << self_play.transposition_table.max_elements << " positions\n";
    self_play.run();
}

void testPosition(torch::jit::script::Module& nnet, torch::Device device) {

    unsigned int time_per_move;

    std::cout << "Enter AI thinking time: ";
    std::cin >> time_per_move;

    std::string start_pos;
    chess::Board startState;
    while (true) {
        std::cout << "Start Position Fen (d for default): ";
        std::cin >> start_pos;

        if (start_pos == "d") {
            start_pos = test_positions[0];
            break;
        }
        try {
            startState = chess::Board(start_pos);
            break;
        }
        catch(...) {
            std::cout << "Invalid Position.\n";
        }
    }

    TranspositionTable<uint64_t, std::pair<std::unordered_map<chess::Move, float>, float>> transposition_table;
    transposition_table.set_size(transposition_table_size);
    std::vector<chess::Board> traversed = {};
    unsigned int num_simulations = 100000, nn_cache_size = 256;

    std::cout << startState << "\n";
    Container container;
    auto rootNode = new Node(container, startState, 0);
    auto newSearch = Search(rootNode, container, traversed, transposition_table, nnet, device, num_simulations, thread_count, nn_cache_size, true, false);
    newSearch.startSearch(true, true, std::chrono::seconds(time_per_move));

    auto white_win_prob = newSearch.getQ()*(1-2*static_cast<int>(startState.sideToMove()));
    auto topLine = newSearch.getTopLine();
    auto move = newSearch.selectMove(true, temperature_end);
    std::cout << "Engine Top Line:\n" << topLine << ", Evaluation = " << probability_to_centipawn(white_win_prob) << '\n';
}

void humanGame(torch::jit::script::Module& nnet, torch::Device device) {

    unsigned int time_per_move, show_tl;

    std::cout << "Enter AI thinking time per move: ";
    std::cin >> time_per_move;

    std::cout << "Show Engine Top Line (enter 0 for no): ";
    std::cin >> show_tl;

    bool tl = static_cast<bool>(show_tl);

    int side = -1;
    bool myTurn;
    while (true) {
        std::cout << "White(0) or Black(1): ";
        std::cin >> side;

        if (side == 0) {
            myTurn = false;
            break;
        }
        else if (side == 1) {
            myTurn = true;
            break;
        }
        else {
            std::cout << "Invalid Option.\n";
        }
    }
    chess::Board startState = chess::Board(test_positions[0]);
    TranspositionTable<uint64_t, std::pair<std::unordered_map<chess::Move, float>, float>> transposition_table;
    transposition_table.set_size(transposition_table_size);
    std::cout << "transposition table size: " << transposition_table.max_elements << " positions\n";
    std::vector<chess::Board> traversed = {};
    unsigned int num_simulations = 10000, nn_cache_size = 256;

    std::cout << startState << "\n";
    uint8_t progress = 0;
    for (int turns = 0; turns < 256; ++turns) {
        while (!myTurn) {
            // basic turn dynamics
            std::cout << "Enter your move: ";
            std::string move;
            std::cin >> move;
            try {
                Move m = uci::parseSan(startState, move);
                if (startState.isCapture(m) || startState.at(m.from()) == chess::PieceType::PAWN) {
                    progress = 0;
                } else {
                    progress += 1;
                }
                startState.makeMove(m);
                traversed.push_back(startState);
                std::cout << startState << "\n";
                myTurn = true;
            }
            catch(...) {
                std::cout << "Invalid Move\n";
            }
        }
        Container container;
        auto rootNode = new Node(container, startState, progress);
        auto newSearch = Search(rootNode, container, traversed, transposition_table, nnet, device, num_simulations, thread_count, nn_cache_size, true, false);
        newSearch.startSearch(true, true, std::chrono::seconds(time_per_move));

        auto white_win_prob = newSearch.getQ()*(1-2*static_cast<int>(startState.sideToMove()));
        auto topLine = newSearch.getTopLine();
        auto move = newSearch.selectMove(true, temperature_end);
        std::cout << "move made: " << move.first << "\n";
        startState.makeMove(move.first);
        std::cout << startState << "\n";
        progress = newSearch.rootNode->moves_since_cpm;
        startState = newSearch.rootNode->state;

        if (tl) { std::cout << "Engine Top Line:\n" << topLine << ", Evaluation = " << probability_to_centipawn(white_win_prob) << '\n'; }
        if (move.second != -1) {
            break;
        }
        myTurn = false;
    }
}

void testGame(torch::jit::script::Module& nnet, torch::jit::script::Module& old_nnet, torch::Device device) {
    
    unsigned int total_games, num_simulations, nn_cache_size;

    std::cout << "Enter the total number of self-play games: ";
    std::cin >> total_games;

    std::cout << "Enter the number of simulations (higher gives more accuracy but longer training time): ";
    std::cin >> num_simulations;

    std::cout << "Enter the eval cache size (higher values will give more breadth but less depth to search and potentially faster training): ";
    std::cin >> nn_cache_size;

    std::string start_pos;
    chess::Board startState;
    while (true) {
        std::cout << "Start Position Fen (d for default): ";
        std::cin >> start_pos;

        if (start_pos == "d") {
            start_pos = test_positions[0];
            break;
        }
        try {
            startState = chess::Board(start_pos);
            break;
        }
        catch(...) {
            std::cout << "Invalid Position.\n";
        }
    }
    auto p1Color = chess::Color::WHITE;
    TranspositionTable<uint64_t, std::pair<std::unordered_map<chess::Move, float>, float>> new_transposition_table;
    TranspositionTable<uint64_t, std::pair<std::unordered_map<chess::Move, float>, float>> old_transposition_table;
    new_transposition_table.set_size(transposition_table_size);
    old_transposition_table.set_size(transposition_table_size);
    std::vector<chess::Board> traversed = {};

    // Directory path where files will be created
    std::string directoryPath = "nnet_test_games";
    std::filesystem::path dir {directoryPath};
    // Create the directory (and any necessary parent directories)
    std::filesystem::create_directories(dir);

    float p1Score = 0;
    float p2Score = 0;

    for (int game = 0; game < total_games; ++game) {

        std::string pgn_moves = "";
        int result = 0;
        bool myTurn = (startState.sideToMove() == p1Color);
        uint8_t progress = 0;
        for (int turns = 0; turns < 256; ++turns) {
            std::pair<chess::Move, int> move;
            if (turns%2 == 0) {
                pgn_moves += std::to_string((turns/2)+1) + ". ";
            }
            if (!myTurn) {
                Container container;
                auto rootNode = new Node(container, startState, progress);
                auto newSearch = Search(rootNode, container, traversed, old_transposition_table, old_nnet, device, num_simulations, thread_count, nn_cache_size, false, false);
                newSearch.startSearch(true);
                std::cout << "P2 Turn, eval = " << probability_to_centipawn(newSearch.getQ()*(1-2*static_cast<int>(startState.sideToMove()))) << ", move - ";
                move = newSearch.selectMove(false, temperature_end, resign_eval_threshold);
                progress = newSearch.rootNode->moves_since_cpm;
            }
            else if (myTurn) {
                Container container;
                auto rootNode = new Node(container, startState, progress);
                auto newSearch = Search(rootNode, container, traversed, new_transposition_table, nnet, device, num_simulations, thread_count, nn_cache_size, false, false);
                newSearch.startSearch(true);
                std::cout << "P1 Turn, eval = " << probability_to_centipawn(newSearch.getQ()*(1-2*static_cast<int>(startState.sideToMove()))) << ", move - ";
                move = newSearch.selectMove(false, temperature_end, resign_eval_threshold);
                progress = newSearch.rootNode->moves_since_cpm;
            }
            if (move.second != -1) {
                if (move.second != 1 && startState.sideToMove() == chess::Color::BLACK) {
                    result = -(move.second - 1);
                }
                else if (move.second != 1 && startState.sideToMove() == chess::Color::WHITE) {
                    result = (move.second - 1);
                }

                if (move.second != 0) {
                    std::cout << move.first << '\n';
                    pgn_moves += chess::uci::moveToSan(startState, move.first) + " ";
                    startState.makeMove(move.first);
                }
                else {
                    std::cout << "resigns.\n";
                }
                break;
            }
            std::cout << move.first << '\n';
            pgn_moves += chess::uci::moveToSan(startState, move.first) + " ";
            startState.makeMove(move.first);
            myTurn = !myTurn;
        }
        std::string result_string = "1/2-1/2";
        p1Score += 0.5;
        p2Score += 0.5;
        if (result == 1) {
            result_string = "1-0";
        }
        else if (result == -1) {
            result_string = "0-1";
        }

        if (p1Color == chess::Color::WHITE) {
            p1Score += result*0.5;
            p2Score -= result*0.5;
        }
        else {
            p1Score -= result*0.5;
            p2Score += result*0.5;
        }

        // Once the game is over, write the PGN data to a file
        std::ofstream pgn_file(directoryPath + "/game_" + std::to_string(game) + ".pgn");

        pgn_file << "[Event \"test game\"]\n";
        pgn_file << "[Site \"?\"]\n";
        pgn_file << "[Date \"????.??.??\"]\n";
        pgn_file << "[Round \"?\"]\n";
        if (p1Color == chess::Color::WHITE) {
            pgn_file << "[White \"New NNet\"]\n";
            pgn_file << "[Black \"Old NNet\"]\n";
        }
        else {
            pgn_file << "[White \"Old NNet\"]\n";
            pgn_file << "[Black \"New NNet\"]\n";
        }
        pgn_file << "[Result \"" + result_string + "\"]\n\n";
        // Write the moves
        pgn_file << pgn_moves << result_string;
        pgn_file.close();
        std::cout << result_string + ", game-" + std::to_string(game) + '\n';
        std::cout << "New Model: " + std::to_string(p1Score) + ", Old Model: " + std::to_string(p2Score) + '\n';
        if (p1Color == chess::Color::WHITE) {p1Color = chess::Color::BLACK;}
        else {p1Color = chess::Color::WHITE;}
        startState = chess::Board(start_pos);
    }
}

std::string findModel(std::string directory_path) {
    std::string model_path;

    if (!std::filesystem::exists(directory_path) || !std::filesystem::is_directory(directory_path)) {
        std::cerr << "Directory does not exist." << std::endl;
    }

    for (const auto& entry : std::filesystem::directory_iterator(directory_path)) {
        const auto& path = entry.path();
        if (std::filesystem::is_regular_file(path)) {
            model_path = path.string();
            break;
        }
    }

    if (model_path.empty()) {
        std::cerr << "No files found in the directory." << std::endl;
    } else {
        std::cout << "Model path set to: " << model_path << std::endl;
    }

    return model_path;
}

int main() {

    ConfigParser parser("params.txt");
    parser.config_params();
    
    std::cout << "Configuration parameters loaded." << std::endl;

    // _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    std::string model_path = findModel(model_directory + "/current_model");
    // Load the model
    torch::jit::script::Module nnet;
    try {
        nnet = torch::jit::load(model_path);
        std::cout << "Model loaded successfully\n";
        // // Print the graph for each method
        // for (const auto& method : nnet.get_methods()) {
        //     std::cout << "Method: " << method.name() << std::endl;
        //     std::cout << method.graph()->toString() << std::endl;
        // }

    } catch (const c10::Error& e) {
        std::cerr << "LibTorch error loading the model: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception while loading the model: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error loading the model\n";
        return -1;
    }

    torch::Device device = torch::kCPU;
    // Check if CUDA is available and move the model to GPU
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available. Moving the model to GPU.\n";
        device = torch::kCUDA;
    } else {
        std::cout << "CUDA not available. Using CPU.\n";
    }

    nnet.to(device, torch::kHalf);
    std::cout << "Model using half precision.\n";

    int choice = -1;
    while (true) {
        std::cout << "Self Play(0), Human Game(1), Test Game(2), or Test Position(3): ";
        std::cin >> choice;

        if (choice == 0) {
            selfPlay(nnet, device);
            break;
        }
        else if (choice == 1) {
            humanGame(nnet, device);
            break;
        }
        else if (choice == 2) {
            std::string old_model_path = findModel(model_directory + "/old_model");
            // Load the model
            torch::jit::script::Module old_nnet;

            try {
                old_nnet = torch::jit::load(old_model_path);
                std::cout << "Model loaded successfully\n";

            } catch (const c10::Error& e) {
                std::cerr << "LibTorch error loading the model: " << e.what() << std::endl;
                return -1;
            } catch (const std::exception& e) {
                std::cerr << "Standard exception while loading the model: " << e.what() << std::endl;
                return -1;
            } catch (...) {
                std::cerr << "Unknown error loading the model\n";
                return -1;
            }

            old_nnet.to(device, torch::kHalf);

            testGame(nnet, old_nnet, device);
            break;
        }
        else if (choice == 3) {
            testPosition(nnet, device);
            break;
        }
        else {
            std::cout << "Invalid Option.\n";
        }
    }

    std::string exitInput;
    std::cout << "Press enter to exit.";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::getline(std::cin, exitInput);
    return 0;
}
