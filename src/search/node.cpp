#include "node.hpp"



Node::Node(Container& container, chess::Board state, chess::Move move, std::vector<Node*> prev_list, float policy)
    : state(state), move(move), prev_list(prev_list), policy(policy) {
        container.push(this);
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
    stateCopy.makeMove(newMove);
    auto new_prevs = prev_list;
    new_prevs.emplace_back(this);
    children.emplace_back(new Node(container, stateCopy, newMove, new_prevs, policy));
}

void Node::backpropagate(float val, Container& container) {
    addToVal(val);
    ++visits;
    virtual_loss = false;

    if (getDepth() > 1) {
        getParent()->backpropagate(-val, container);
    }
}
