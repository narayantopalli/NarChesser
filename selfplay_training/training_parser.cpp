#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <filesystem>
#include <cmath>
#include "../src/chess.hpp"
#include "training_planes.hpp"
#include "train_policy_map.hpp"

using namespace chess;

class MyVisitor : public pgn::Visitor {
   public:
    virtual ~MyVisitor() {}

    void startPgn() { board.setFen(constants::STARTPOS); boards.push_back(board);}

    void header(std::string_view key, std::string_view value) {}

    void startMoves() {}

    void move(std::string_view move, std::string_view comment) override {
        Move m = uci::parseSan(board, move);
        moves.push_back(m);
        board.makeMove(m);
        gameTurns += 1;
        boards.push_back(board);

    }

    void endPgn() {}
    
    int gameTurns = 0;
    Board board;
    std::vector<Board> boards;
    std::vector<Move> moves;
};

void process_value_file(const std::filesystem::path& path, std::vector<float>& values);
void process_policy_file(const std::filesystem::path& policy_file_path, const std::string& output_file_path);
void process_game_file(const std::filesystem::path& path, const std::vector<float>& values);

// Main function to parse game data
void parse_game(const std::filesystem::path& path) {
    std::vector<float> values;
    process_value_file(path / "q_values.bin", values);
    process_policy_file(path / "policy.bin", "policy_labels.bin");
    process_game_file(path, values);
}

// Processing value file
void process_value_file(const std::filesystem::path& value_file_path, std::vector<float>& values) {
    std::ifstream value_file_stream(value_file_path, std::ios::binary);
    if (!value_file_stream) {
        std::cerr << "Failed to open value file\n";
        return;
    }

    float value;
    while (value_file_stream.read(reinterpret_cast<char*>(&value), sizeof(float))) {
        values.push_back(value);
    }
}

// Processing policy file
void process_policy_file(const std::filesystem::path& policy_file_path, const std::string& output_file_path) {
    std::ifstream policy_file_stream(policy_file_path, std::ios::binary);
    if (!policy_file_stream.is_open()) {
        std::cerr << "Failed to open policy file: " << policy_file_path << std::endl;
        return;
    }

    std::ofstream PolicyLabelsFile(output_file_path, std::ios::binary | std::ios::app);
    if (!PolicyLabelsFile.is_open()) {
        std::cerr << "Failed to open output file: " << output_file_path << std::endl;
        return;
    }

    constexpr size_t BUFFER_SIZE = 49152;
    char buffer[BUFFER_SIZE];

    while (policy_file_stream.read(buffer, BUFFER_SIZE)) {
        PolicyLabelsFile.write(buffer, policy_file_stream.gcount());
    }
    // Write any remaining part of the file
    PolicyLabelsFile.write(buffer, policy_file_stream.gcount());

    if (policy_file_stream.bad()) {
        std::cerr << "I/O error while reading from policy file." << std::endl;
    }
    if (PolicyLabelsFile.bad()) {
        std::cerr << "I/O error while writing to output file." << std::endl;
    }

    // Close file streams
    policy_file_stream.close();
    PolicyLabelsFile.close();
}

// Processing game file and exporting data
void process_game_file(const std::filesystem::path& path, const std::vector<float>& values) {
    const auto file = path / "game.pgn";
    std::ifstream file_stream(file);
    if (!file_stream) {
        std::cerr << "Failed to open game file: " << file << std::endl;
        return;
    }

    auto vis = std::make_unique<MyVisitor>();
    pgn::StreamParser parser(file_stream);
    parser.readGames(*vis);

    std::ofstream value_labels_file("value_labels.bin", std::ios::binary | std::ios::app);
    std::ofstream data_file("encoded_states.bin", std::ios::binary | std::ios::app);
    std::ofstream game_fens_file("selfplay_game_fens.csv", std::ios::app);

    // Process each turn in the game
    for (int i = 0; i < (vis->gameTurns); ++i) {
        std::unique_ptr<std::unique_ptr<Bitboard[]>[]> encodedState = std::make_unique<std::unique_ptr<Bitboard[]>[]>(5);
        for (int lookBack = 3; lookBack >= 0; lookBack--) {
            if ((i - lookBack) < 0) {
                encodedState[lookBack] = planes::emptyPlane();
            }
            else {
                encodedState[lookBack] = planes::toPlane(vis->boards[(i - lookBack)], vis->boards[i].sideToMove());
            }
        }

        encodedState[4] = planes::extraPlanes(vis->boards[i]);
        
        // Export to binary.
        for (int h = 0; h < 4; ++h) {
            for (int k = 0; k < 14; ++k) {
                data_file.write(reinterpret_cast<const char*>(&encodedState[h][k]), sizeof(Bitboard));
            }
        }
        for (int l = 0; l < 8; ++l) {
            data_file.write(reinterpret_cast<const char*>(&encodedState[4][l]), sizeof(Bitboard));
        }

        int res = color_to_result(vis->boards[i].sideToMove(), vis->game_result);

        float temperature;
        if (i < 30) {temperature = 0.75; }
        else if (i >= 30) {temperature = 0.1; }

        float epsilon = pow(0.3, temperature);
        float val = epsilon*static_cast<float>(res) + (1-epsilon)*values[i];
        value_labels_file.write(reinterpret_cast<const char*>(&val), sizeof(float));

        game_fens_file << vis->boards[i].getFen() + ",";

        value_labels_file.flush();
        data_file.flush();
        game_fens_file.flush();
    }

    value_labels_file.close();
    data_file.close();
    game_fens_file.close();
}

int main(int argc, char const *argv[]) {

    std::string base_path = "../selfplay_training/selfplay_training_games"; // directory path

    try {
        // Check if the base path exists and is a directory
        if (std::filesystem::exists(base_path) && std::filesystem::is_directory(base_path)) {
            for (const auto& first_level_entry : std::filesystem::directory_iterator(base_path)) {
                if (std::filesystem::is_directory(first_level_entry.status())) {
                    int count = 0;
                    for (const auto& second_level_entry : std::filesystem::directory_iterator(first_level_entry.path())) {
                        if (std::filesystem::is_directory(second_level_entry.status())) {

                            parse_game(second_level_entry.path());
                            count++;
                        }
                    }
                    std::cout << "Processed " << count << " games in directory: " << first_level_entry.path() << std::endl;
                }
            }
            std::cout << "Done processing all folders." << std::endl;
        } else {
            std::cerr << "Provided path is not a directory or does not exist: " << base_path << std::endl;
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }

    std::cout << "done with all folders";
    return 0;

}
