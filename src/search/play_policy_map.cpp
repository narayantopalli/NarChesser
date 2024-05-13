#include "play_policy_map.hpp"
#include "../functions.hpp"

std::unique_ptr<float[]> policy_map::get_move_to_policy(std::unordered_map<chess::Move, float>& move_map, chess::Color color) {

    int c = static_cast<int>(color);
    const int totalSize = PLANES * BOARD_SIZE * BOARD_SIZE;
    auto policyIndex = std::make_unique<float[]>(totalSize);
    for (int i = 0; i < totalSize; ++i) {
        policyIndex[i] = 0.0f;
    }

    for (const auto& move : move_map) {
        int from_rank_index = (c == 0 ? static_cast<int>(move.first.from().rank()) : 7 - static_cast<int>(move.first.from().rank()));
        int from_file_index = static_cast<int>(move.first.from().file());
        int dest_rank_index = (c == 0 ? static_cast<int>(move.first.to().rank()) : 7 - static_cast<int>(move.first.to().rank()));
        int dest_file_index = static_cast<int>(move.first.to().file());
        int promotion_to_int = promotion_to_index(move.first.promotionType());
        for (int plane = 0; plane < PLANES; ++plane) {
            bool rightSquare = (policyMap[(from_rank_index*BOARD_SIZE*PLANES) + (from_file_index*PLANES) + plane] == ((BOARD_SIZE*dest_rank_index) + dest_file_index)*promotion_to_int);
            if (rightSquare) {
                policyIndex[(from_rank_index*BOARD_SIZE*PLANES) + (from_file_index*PLANES) + plane] = move.second;
                break;
            }
        }
    }
    return policyIndex;
}

std::unordered_map<chess::Move, float> policy_map::policy_to_moves(const std::vector<float> policy, const chess::Board& state, bool tactic_bonus) {
    int c = static_cast<int>(state.sideToMove());
    std::unordered_map<chess::Move, float> move_map;
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, state);

    float max_policy = std::numeric_limits<float>::lowest();
    float tactic_total = 0.0f;
    std::vector<chess::Move> tactics = {};
    for (int i = 0; i < moves.size(); ++i) {
        const auto move = moves[i];
        int from_rank_index = (c == 0 ? static_cast<int>(move.from().rank()) : 7 - static_cast<int>(move.from().rank()));
        int from_file_index = static_cast<int>(move.from().file());
        int dest_rank_index = (c == 0 ? static_cast<int>(move.to().rank()) : 7 - static_cast<int>(move.to().rank()));
        int dest_file_index = static_cast<int>(move.to().file());
        int promotion_to_int = promotion_to_index(move.promotionType());
        for (int plane = 0; plane < PLANES; ++plane) {
            bool rightSquare = (policyMap[(from_rank_index*BOARD_SIZE*PLANES) + (from_file_index*PLANES) + plane] == ((BOARD_SIZE*dest_rank_index) + dest_file_index)*promotion_to_int);
            if (rightSquare) {
                auto pi_i = policy[(from_rank_index*BOARD_SIZE*PLANES) + (from_file_index*PLANES) + plane];
                if (pi_i > max_policy) { max_policy = pi_i; }
                if (tactic_bonus) {
                    bool tactic_check = false;
                    auto stateCopy = state;
                    if (state.isCapture(move)) { tactic_check = true; }
                    stateCopy.makeMove(move);
                    if (stateCopy.inCheck()) { tactic_check = true; }
                    if (tactic_check) { 
                        tactics.push_back(move); 
                        tactic_total += pi_i;
                    }
                }
                move_map.insert({move, pi_i});
                break;
            }
        }
    }
    if (tactic_bonus && !tactics.empty()) {
        // tactics should average around max_policy
        auto c = max_policy - (tactic_total/static_cast<float>(tactics.size()));
        for (const auto& m : tactics) {
            move_map[m] += c;
        }
    }
    return Softmax(move_map);
}
