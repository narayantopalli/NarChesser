#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <filesystem>
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

void parse_game(std::filesystem::path path) {

    //sending to binary
    std::ofstream DataFile("encoded_states.bin", std::ios::binary | std::ios::app);
    std::ofstream PolicyLabelsFile("policy_labels.bin", std::ios::binary | std::ios::app);
    std::ofstream ValueLabelsFile("value_labels.bin", std::ios::binary | std::ios::app);

    // int success = 0;
    char DataBuffer[49152];
    DataFile.rdbuf()->pubsetbuf(DataBuffer, sizeof(DataBuffer));
    char PolicyBuffer[49152];
    PolicyLabelsFile.rdbuf()->pubsetbuf(PolicyBuffer, sizeof(PolicyBuffer));
    char ValueBuffer[4096];
    ValueLabelsFile.rdbuf()->pubsetbuf(ValueBuffer, sizeof(ValueBuffer));

    const auto file  = path;
    auto file_stream = std::ifstream(file);

    if (!file_stream.is_open()) {
    std::cerr << "Failed to open file: " << file << std::endl;
    return;
    }

    auto vis = std::make_unique<MyVisitor>();

    pgn::StreamParser parser(file_stream);
    parser.readGames(*vis);

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
                DataFile.write(reinterpret_cast<const char*>(&encodedState[h][k]), sizeof(Bitboard));
                // success+=1;
            }
        }
        for (int l = 0; l < 8; ++l) {
            DataFile.write(reinterpret_cast<const char*>(&encodedState[4][l]), sizeof(Bitboard));
            // success+=1;
        }

        int res = color_to_result(vis->boards[i].sideToMove(), vis->game_result);
        // success+=1;
        ValueLabelsFile.write(reinterpret_cast<const char*>(&res), sizeof(int));
        auto index = get_move_to_policy(vis->moves[i], vis->boards[i].sideToMove());
        // std::cout << vis->moves[i] << vis->boards[i].sideToMove();
        // Calculate the number of uint64_t chunks needed
        const int numChunks = 73;
        for (int chunkIndex = 0; chunkIndex < numChunks; ++chunkIndex) {
            uint64_t chunk = 0;
            for (int bitIndex = 0; bitIndex < 64; ++bitIndex) {
                if (index.test(chunkIndex * 64 + bitIndex)) {
                    chunk |= (uint64_t(1) << bitIndex);
                }
            }
            // Write the uint64_t chunk to the file
            PolicyLabelsFile.write(reinterpret_cast<const char*>(&chunk), sizeof(uint64_t));
        }

        DataFile.flush();
        ValueLabelsFile.flush();
        PolicyLabelsFile.flush();
    }
    DataFile.close();
    PolicyLabelsFile.close();
    ValueLabelsFile.close();
}

int main(int argc, char const *argv[]) {

    std::string path = "../training/training_games"; // directory path

    try {
        if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
            for (const auto& entry : std::filesystem::directory_iterator(path)) {
                if (std::filesystem::is_directory(entry.status())) {
                    int count = 0;
                    for (const auto& f : std::filesystem::directory_iterator(entry.path())) {

                        auto filename = f.path().filename(); // Get the filename
                        if (std::filesystem::is_regular_file(f.status())) { // Check if it's a file
                            std::cout << filename << std::endl;
                            auto full_path = entry.path() / f.path().filename(); // Get the full file path
                            parse_game(full_path);
                            count +=1;
                        }
                        std::cout << count << std::endl;
                    }
                }
                std::cout << "done with folder";
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << e.what() << std::endl;
    }

    std::cout << "done with all folders";
    return 0;

}
