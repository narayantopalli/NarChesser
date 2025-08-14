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
    std::atomic<float> val_sum = 0.0f;
    uint8_t moves_since_cpm;
    float progress_mult = 1.0f;
    // bool check_or_cap;
    chess::Move move;
    chess::Board state;
    
    std::mutex lock;
    std::mutex expand_lock;
    std::atomic<bool> in_nnet = false;
    std::atomic<bool> virtual_loss = false;
    std::condition_variable nnet_cv;


    inline Node* getParent() const;
    inline uint8_t getDepth() const;
    Node(Container& container, chess::Board state, uint8_t moves_since_cpm, chess::Move move = chess::Move::NULL_MOVE, std::vector<Node*> prev_list = {}, float policy = 0.0f);
    inline float puct_value(const float v_loss_c = 1.0f);
    inline bool is_leaf_node() const;
    std::pair<bool, float> get_terminal_val() const;
    void expand(chess::Move newMove, float policy, Container& container);
    inline void addToVal(float val);
    inline float cpmToMult(const uint8_t moves_since_cpm) const;
    void backpropagate(float val, Container& container);
    inline float getQ(const float v_loss = 0.0f) const;
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

inline float Node::puct_value(const float v_loss_c /* = 1.0f */) {
    const int n = visits.load(std::memory_order_relaxed);
    const bool vloss = virtual_loss.load(std::memory_order_relaxed);

    // Unvisited nodes: encourage first visit unless someone is already “in flight”
    if (n == 0) {
        return vloss ? -std::numeric_limits<float>::infinity()
                     :  std::numeric_limits<float>::infinity();
    }

    const Node* parent = getParent();
    if (parent == nullptr) {
        std::cout << "Parent is nullptr\n";
        return std::numeric_limits<float>::infinity();
    }
    const int parent_n = parent->visits.load(std::memory_order_relaxed);

    // Treat virtual loss as extra temporary visits on this edge only
    const float v_loss = vloss ? v_loss_c : 0.0f;
    const float denom  = 1.0f + static_cast<float>(n) + v_loss;

    // U term uses parent count (matches sqrt(N_parent) form)
    const float U = cpuct(parent_n) * policy *
                    std::sqrt(static_cast<float>(parent_n)) / denom;

    return getQ(v_loss) * progress_mult + U;
}

inline bool Node::is_leaf_node() const {
    return children.empty();
}

inline void Node::addToVal(const float val) {
    val_sum.fetch_add(val, std::memory_order_relaxed);
}

/*
    @return Progress multiplier based on moves since Capture/Pawn Move
    @param moves_since_cpm: Number of moves since the last Capture/Pawn Move
*/
inline float Node::cpmToMult(const uint8_t moves_since_cpm) const {
    uint8_t interval = 100 - std::min(100, static_cast<int>(moves_since_cpm));
    return 1.0/(1.0 + exp(0.08*(25.0-static_cast<float>(interval))));
}

/*
    @return Q value of the node
    @param v_loss: Virtual loss
*/
inline float Node::getQ(const float v_loss) const {
    const int n = visits.load(std::memory_order_relaxed);
    if (n <= 0) return 0.0f;
    const float w = val_sum.load(std::memory_order_relaxed);
    return w/(n + v_loss);
}
