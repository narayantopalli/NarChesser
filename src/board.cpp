// #include <iostream>

// #include "chess.hpp"
// #include "manipulators.hpp"
// #include "planes.hpp"

// using namespace chess;

// uint64_t perft(Board& board, int depth) {
//     Movelist moves;
//     movegen::legalmoves(moves, board);

//     if (depth == 1) {
//         return moves.size();
//     }

//     uint64_t nodes = 0;

//     for (int i = 0; i < moves.size(); i++) {
//         const auto move = moves[i];
//         board.makeMove(move);
//         nodes += perft(board, depth - 1);
//         board.unmakeMove(move);
//     }

//     return nodes;
// }

// int main() {
//     Board board = Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

//     std::cout << perft(board, 5) << std::endl;

//     return 0;
// }