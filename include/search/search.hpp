#pragma once

#include <map>
#include <thread>
#include <cmath>
#include "node.hpp"
#include "play_policy_map.hpp"
#include "transposition_table.hpp"
#include "threadpool.hpp"
#include "include/utils/functions.hpp"
#include "include/model/encoder.hpp"
#include "include/model/model.hpp"

class Search {

    public:
    unsigned int num_threads;
    unsigned int num_simulations;
    uint8_t max_depth = 0;
    std::mutex depth_lock;
    uint32_t total_nodes = 1;
    Node* rootNode = nullptr;
    torch::jit::script::Module& nnet;
    torch::Device device;
    Container& container;
    std::vector<chess::Board>& traversed;
    TranspositionTable<uint64_t, std::pair<std::unordered_map<chess::Move, float>, float>>& transposition_table;
    unsigned int nn_batch_size;
    const int policySize = PLANES * BOARD_SIZE * BOARD_SIZE;
    bool depthVerbose;
    const uint8_t position_history;

    bool evaluating = false;
    std::condition_variable evaluating_cv;

    Search(Node* rootNode, Container& container, std::vector<chess::Board>& traversed, TranspositionTable<uint64_t, std::pair<std::unordered_map<chess::Move, float>, float>>& transposition_table, 
        torch::jit::script::Module& nnet, torch::Device device, unsigned int num_simulations, 
        unsigned int num_threads, unsigned int nn_batch_size, bool depthVerbose = false, const uint8_t position_history = 1);
    chess::Movelist get_moves(const chess::Board& state) const;
    void evaluate_nodes();
    void expand_leaf(Node* node, std::unique_lock<std::mutex> lock);
    void expandRoot(Node* root, const bool noise);
    void expand(Node* node);
    void move_root(const Node* newRoot);
    std::pair<chess::Move, int> selectMove(const bool verbose, double temperature, float resign_threshold = 1.0);
    void makeMove(const chess::Move m);
    void nn_eval_request(const unsigned int batch_size);
    std::pair<std::pair<torch::Tensor, torch::Tensor>, Node*> get_evaluation();
    float getRootQ() const;
    std::string getTopLine();
    inline void checkMaxDepth(const uint8_t depth);
    inline void startSearch(const bool dirichelet_noise, bool use_time = false, std::chrono::duration<int> const& max_time = std::chrono::seconds(0));
    inline void pushToCache(torch::Tensor state_tensor, Node* node);

    private:
    std::mutex cache_lock;
    std::mutex request_guard;
    std::mutex eval_guard;
    std::queue<torch::Tensor> nn_cache;
    std::queue<Node*> nn_address_cache;
    std::queue<std::pair<torch::Tensor, torch::Tensor>> nn_evaluations;

    struct ThreadManager {

        Search& search;
        ThreadManager(Search& search) : search(search) {};
        void startSearch(const bool dirichelet_noise, bool use_time, std::chrono::duration<int> const& max_time);
        void workerSearch();
        void threadEvaluation();
        void evaluate();
        void evaluateRoot(const bool noise);
        bool already_started = false;

        std::atomic<int> waiting_threads = 0;

    };

    ThreadManager threadManager;
};

inline void Search::pushToCache(torch::Tensor state_tensor, Node* node) {
    std::unique_lock<std::mutex> guard(cache_lock);
    while (evaluating) evaluating_cv.wait(guard);
    nn_cache.push(state_tensor);
    nn_address_cache.push(node);
    guard.unlock();
}

inline void Search::checkMaxDepth(const uint8_t depth) {
    std::lock_guard<std::mutex> guard(depth_lock);
    if (depth > max_depth) {
        max_depth = depth;
        std::cout << "\rDEPTH: " << static_cast<unsigned int>(max_depth) << ", NODES: " << container.size() << ", TTF: " << std::ceil(10000*static_cast<float>(transposition_table.size())/static_cast<float>(transposition_table.max_elements))/100 << "%" << std::flush;
    }
}

inline void Search::startSearch(const bool dirichelet_noise, bool use_time, std::chrono::duration<int> const& max_time) {
    max_depth = 0;
    threadManager.startSearch(dirichelet_noise, use_time, max_time);
    nn_eval_request(1);
}

inline float Search::getRootQ() const {
    auto total_visits = 0;
    float total_value = 0.0f;
    for (const auto& child : rootNode->children) {
        total_visits += child->visits.load();
        total_value += child->val_sum.load();
    }
    return total_value/static_cast<float>(total_visits);
}
