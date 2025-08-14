#include "node.hpp"



Node::Node(Container& container, chess::Board state, uint8_t moves_since_cpm, chess::Move move, std::vector<Node*> prev_list, float policy)
    : state(state), move(move), prev_list(prev_list), policy(policy), moves_since_cpm(moves_since_cpm) {
        container.push(this);
        progress_mult = cpmToMult(moves_since_cpm);
    }

std::pair<bool, float> Node::get_terminal_val() const {
    float val;
    auto check = state.isGameOver();
    if (check.second != chess::GameResult::NONE) {
        if (check.second == chess::GameResult::DRAW) {
            val = 0.0;
        } else if (check.second == chess::GameResult::LOSE) {
            val = 1.0;
        }
        return std::make_pair(true, val);
    }
    return std::make_pair(false, 0.0);
}

void Node::expand(chess::Move newMove, float policy, Container& container) {
    std::lock_guard<std::mutex> guard(expand_lock);
    auto stateCopy = state;
    uint8_t progress;
    // bool check_or_cap = false;
    if (stateCopy.isCapture(newMove)) {
        progress = 0;
        // check_or_cap = true;
    } 
    else if (stateCopy.at(newMove.from()) == chess::PieceType::PAWN) {
        progress = 0;
    }
    else {
        progress = moves_since_cpm + 1;
    }

    stateCopy.makeMove(newMove);
    // check_or_cap = stateCopy.inCheck();
    auto new_prevs = prev_list;
    new_prevs.emplace_back(this);
    children.emplace_back(new Node(container, stateCopy, progress, newMove, new_prevs, policy));
}

void Node::backpropagate(float val, Container& container) {
    addToVal(val);
    ++visits;
    virtual_loss = false;

    if (getDepth() > 1) {
        getParent()->backpropagate(-val, container);
    }
}
