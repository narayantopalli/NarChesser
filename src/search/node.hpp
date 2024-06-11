#pragma once

#include <memory>
#include <vector>
#include <list>
#include <iostream>
#include <cmath>
#include <utility>
#include <limits>
#include <mutex>
#include <atomic>
#include <queue>
#include <algorithm>
#include <condition_variable>
#include "constants.hpp"
#include "../chess.hpp"
#include "../planes.hpp"

class Node;
class Container;

struct Node {

    std::vector<Node*> children = {};
    std::vector<Node*> prev_list = {};
    float policy = 0.0f;
    std::atomic<int> visits = 0;
    float value = 0.0f;
    uint8_t moves_since_cpm;
    float progress_mult = 1.0f;
    chess::Move move;
    chess::Board state;
    std::mutex lock;
    std::mutex add_lock;
    std::mutex expand_lock;
    std::atomic<bool> in_nnet = false;
    std::atomic<bool> virtual_loss = false;
    std::condition_variable nnet_cv;


    inline Node* getParent() const;
    inline uint8_t getDepth() const;
    Node(Container& container, chess::Board state, uint8_t moves_since_cpm, chess::Move move = chess::Move::NULL_MOVE, std::vector<Node*> prev_list = {}, float policy = 0.0f);
    inline float puct_value(float v_loss_c = 1.0f);
    inline bool is_leaf_node() const;
    std::pair<bool, float> get_terminal_val() const;
    void expand(chess::Move newMove, float policy, Container& container);
    inline void addToVal(float val);
    inline float cpmToMult(const uint8_t moves_since_cpm) const;
    void backpropagate(float val, Container& container);

};


struct Container {

    friend class Search;

    public:
        Container() {}

        ~Container() {
            for (auto& n :list) {
                delete n;
            }
            // std::cout << "Container Freed\n";
        }

        void push(Node* node) {
            std::lock_guard<std::mutex> guard(lock);
            list.push_back(node);
        }

        std::list<Node*>::iterator removeNode(const std::list<Node*>::iterator it) {
            delete *it;
            *it = nullptr;
            std::list<Node*>::iterator nextIt = list.erase(it);
            return nextIt;
        }

        uint32_t size() const {
            return list.size();
        }

        bool empty() const {
            return !list.size();
        }

        Node* front() const {
            return list.front();
        }

        Node* back() const {
            return list.back();
        }

        std::list<Node*> list;

    private:
        std::mutex lock;
};

inline Node* Node::getParent() const {
    if (!prev_list.empty()) return prev_list.back();
    return nullptr;
}

inline uint8_t Node::getDepth() const {
    return prev_list.size();
}

inline float Node::puct_value(float v_loss_c) {
    if (visits == 0) {
        if (virtual_loss.load()) {
            return std::numeric_limits<float>::lowest();
        }
        return std::numeric_limits<float>::max();
    }

    float ucb = cpuct(visits) * policy * std::sqrt(static_cast<float>(getParent()->visits)) / (1 + visits);

    return ((value*progress_mult - v_loss_c * virtual_loss.load())/ visits) + ucb;
}

inline bool Node::is_leaf_node() const {
    return children.empty();
}

inline void Node::addToVal(float val) {
    std::lock_guard<std::mutex> guard(add_lock);
    value+=val;
}

inline float Node::cpmToMult(const uint8_t moves_since_cpm) const {
    uint8_t interval = 100 - std::min(100, static_cast<int>(moves_since_cpm));
    return 1.0/(1.0 + exp(0.08*(25.0-static_cast<float>(interval))));
}
