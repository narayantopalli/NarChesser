#include "search.hpp"

// Constructor for the Search class. It initializes an instance of the class with the necessary components for performing a search in a chess game. 
// These components include nodes, containers for storing game states, a transposition table for memoization, a neural network model, 
// and various configuration parameters like the number of simulations and threads to use.
Search::Search(Node* rootNode, Container& container, std::vector<chess::Board>& traversed,
               TranspositionTable<uint64_t, std::pair<std::unordered_map<chess::Move, float>, float>>& transposition_table, 
               torch::jit::script::Module& nnet, torch::Device device, unsigned int num_simulations, 
               unsigned int num_threads, unsigned int nn_batch_size, bool depthVerbose, bool tactic_bonus, const uint8_t position_history)
    : rootNode(rootNode), container(container), traversed(traversed), transposition_table(transposition_table), 
      nnet(nnet), device(device), num_simulations(num_simulations + 1), num_threads(num_threads), 
      nn_batch_size(nn_batch_size), threadManager(*this), depthVerbose(depthVerbose), tactic_bonus(tactic_bonus), position_history(position_history) {}

// Retrieves all legal chess moves for a given board state. This is used to determine possible next moves from any given position.
chess::Movelist Search::get_moves(const chess::Board& state) const {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, state);
    return moves;
}

// Expands a leaf node in the search tree using the neural network to evaluate the position. 
// It also updates the transposition table to avoid re-evaluating the same positions.
void Search::expand_leaf(Node* node, std::unique_lock<std::mutex> lock) {
    // Initialization of the neural network evaluation structure
    std::pair<std::unordered_map<chess::Move, float>, float> nn_eval;
    auto state_hash = node->state.hash();
    if (transposition_table.contains(state_hash)) {
        // If evaluation exists, use it to expand the node with possible moves
        nn_eval = transposition_table.getHash(state_hash);
        auto movelist = get_moves(node->state);
        for (const auto &move : movelist) {
            node->expand(move, nn_eval.first[move], container);
        }
        node->in_nnet = false;
        node->nnet_cv.notify_all();
        lock.unlock();
        node->backpropagate(nn_eval.second, container);
        if (depthVerbose) {checkMaxDepth(node->getDepth());}
    } else {
        // Otherwise, mark the node for neural network evaluation
        lock.unlock();
        if (depthVerbose) {checkMaxDepth(node->getDepth());}
        pushToCache(EncodedState(node, traversed, position_history).toTensor(), node);
        nn_eval_request(nn_batch_size);
    }
}

// Expands the root node of the search tree, optionally applying Dirichlet noise for exploration enhancement.
void Search::expandRoot(Node* root, const bool noise) {

    std::unique_lock<std::mutex> guard(root->lock);
    std::pair<std::unordered_map<chess::Move, float>, float> nn_eval;

    auto state_hash = root->state.hash();
    if (transposition_table.contains(state_hash)) {
        nn_eval = transposition_table.getHash(state_hash);
        auto movelist = get_moves(root->state);
        auto policy = noise ? applyDirichletNoise(nn_eval.first, root_dirichlet_alpha, root_dirichlet_epsilon) : nn_eval.first;
        for (const auto &move : movelist) {
            root->expand(move, policy[move], container);
        }
        guard.unlock();
    } else {
        pushToCache(EncodedState(root, traversed, position_history).toTensor(), root);
        auto evaluation = model::evaluate(nn_cache, nnet, device);
        nn_evaluations = evaluation;
        threadManager.evaluateRoot(noise);
    }
}

// Recursive method for expanding nodes starting from a specific node. It selects the best node to expand based on a heuristic.
void Search::expand(Node* node) {
    Node* selection = nullptr;
    auto terminal = node->get_terminal_val();

    std::unique_lock<std::mutex> guard(node->lock);
    while (node->in_nnet.load()) node->nnet_cv.wait(guard);

    if (terminal.first) {
        guard.unlock();
        // If the node represents a terminal state (game over), backpropagate the result
        node->backpropagate(terminal.second, container);
        if (depthVerbose) {checkMaxDepth(node->getDepth());}
    } else {
        if (node->is_leaf_node()) {
            // If it's a leaf node, try to expand it
            expand_leaf(node, std::move(guard));
        } else {
            // For internal nodes, select the best child based on a score and recursively expand it
            float highest_puct = std::numeric_limits<float>::lowest();
            for (const auto& child : node->children) {
                float child_val = child->puct_value();
                if (child_val > highest_puct) {
                    highest_puct = child_val;
                    selection = child;
                }
            }

            // Ensure a proper selection and avoid bottlenecks
            if (selection == nullptr) {
                nn_eval_request(1);
                for (const auto& child : node->children) {
                    float child_val = child->puct_value();
                    if (child_val > highest_puct) {
                        highest_puct = child_val;
                        selection = child;
                    }
                }
                if (selection == nullptr) {
                    selection = node->children.front(); // safe selection set, helps avoid bugs especially in positions with few moves
                }
            }
            selection->virtual_loss = true;
            guard.unlock();
            expand(selection); // Recursively expand the selected node
        }
    }
}

// Adjusts the root of the search tree based on the current game state. This involves moving nodes around to reflect the game's progression.
void Search::move_root(const Node* newRoot) {

    // Move the old root to the traversed container
    ++total_nodes;
    auto it = container.list.begin();
    Node* rootNode = *it;
    chess::Board rootState = rootNode->state;
    traversed.push_back(rootState);
    it = container.removeNode(it);

    // Clean up the tree by removing nodes that are not in the path to the new root
    while (it != container.list.end()) {
        ++total_nodes;
        Node* node = *it;
        if (!((node->getDepth() == 1 && node == newRoot) || (node->getDepth() > 1 && node->prev_list[1] == newRoot))) {
            it = container.removeNode(it);
        } else {
            if (!node->prev_list.empty()) {
                node->prev_list.erase(node->prev_list.begin());
            }
            ++it;
        }
    }
}

// Selects the next move based on the visit counts of the children of the root node, applying a temperature parameter to influence the selection.
std::pair<chess::Move, int> Search::selectMove(const bool verbose, double temperature, float resign_threshold) {
    uint16_t highest_visit_count = 0;
    Node* selection;
    std::vector<Node*> nodes = {};
    std::vector<float> probabilities = {};
    for (const auto &child : rootNode->children) {
        // verbose turns on move policy and value outputs to console
        if (verbose) {
            // for debugging
            int c = static_cast<int>(rootNode->state.sideToMove());
            int from_rank_index = (c == 0 ? static_cast<int>(child->move.from().rank()) : 7 - static_cast<int>(child->move.from().rank()));
            int from_file_index = static_cast<int>(child->move.from().file());
            int dest_rank_index = (c == 0 ? static_cast<int>(child->move.to().rank()) : 7 - static_cast<int>(child->move.to().rank()));
            int dest_file_index = static_cast<int>(child->move.to().file());
            int promotion_to_int = promotion_to_index(child->move.promotionType());
            for (int plane = 0; plane < PLANES; ++plane) {
                bool rightSquare = (policyMap[(from_rank_index*BOARD_SIZE*PLANES) + (from_file_index*PLANES) + plane] == ((BOARD_SIZE*dest_rank_index) + dest_file_index)*promotion_to_int);
                if (rightSquare) {
                    std::cout << "Policy Index: " << (from_rank_index*BOARD_SIZE*PLANES) + (from_file_index*PLANES) + plane << ", ";
                    break;
                }
            }
            std::cout << "Move: " << child->move << ", Visits: " << child->visits << ", Policy: " << child->policy << ", Value: " << child->value/child->visits << ", PUCT Value: " << child->puct_value() << '\n';
        }
        nodes.push_back(child);
        probabilities.push_back(std::pow((static_cast<long double>(child->visits.load())/static_cast<long double>(num_simulations)), static_cast<long double>(1/temperature)));
    }
    // Use a random distribution to select a node based on the computed probabilities
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> distr(probabilities.begin(), probabilities.end());
    selection = nodes[distr(gen)];

    // Get game result (-1 if no result yet)
    int result = -1;
    if ((selection->value/selection->visits) < -resign_threshold) {
        if (getQ() < -resign_threshold) {result = 0;}
    }
    else {
        if (selection->state.isGameOver().second == chess::GameResult::LOSE) {result = 2;}
        else if (selection->state.isGameOver().second == chess::GameResult::DRAW) {result = 1;}
    }

    // Move the root of the search tree to the selected node
    move_root(selection);
    rootNode = selection;
    return std::make_pair(selection->move, result);
}

// Updates the tree's root to reflect a move made in the game, essentially progressing the game state.
void Search::makeMove(const chess::Move m) {
    Node* selection;
    for (const auto &child : rootNode->children) {
        if (child->move == m) {
            selection = child;
            break;
        }
    }
    // Move the root of the tree to the selected state
    move_root(selection);
    rootNode = selection;
}

// Adds a node's state to the queue for neural network evaluation, batching evaluations to improve efficiency.
void Search::nn_eval_request(const unsigned int batch_size) {
    
    std::unique_lock<std::mutex> request(request_guard);
    while (evaluating) evaluating_cv.wait(request);

    if (nn_cache.size() >= batch_size) {
        // std::cout << "need to evaluate\n";
        evaluating = true;
        evaluating_cv.notify_all();
        evaluate_nodes();
        // std::cout << evaluating << ", " << nn_cache.size() << '\n';
        evaluating = false;
        evaluating_cv.notify_all();
        request.unlock();
    }
    else {
        request.unlock();
    }
}

// Processes the neural network evaluations for the nodes in the batch, updating them with the results.
void Search::evaluate_nodes() {
    auto evaluation = model::evaluate(nn_cache, nnet, device);
    nn_evaluations = evaluation;
    threadManager.threadEvaluation();
}

// Retrieves a neural network evaluation result for a specific node.
std::pair<std::pair<torch::Tensor, torch::Tensor>, Node*> Search::get_evaluation() {
    std::unique_lock<std::mutex> eval(eval_guard);
    auto node = nn_address_cache.front();
    auto evaluation_tensor = nn_evaluations.front();

    nn_address_cache.pop();
    nn_cache.pop();
    nn_evaluations.pop();

    return std::make_pair(evaluation_tensor, node);
}

std::string Search::getTopLine() {
    auto sel = rootNode;
    std::string topLine = "";
    int i = 1;
    if (sel->state.sideToMove() == chess::Color::BLACK) {
        topLine += "1... ";
        ++i;
    }
    while (!sel->children.empty()) {
        float highest_val = std::numeric_limits<float>::lowest();
        for (const auto& child : sel->children) {
            float child_val = child->visits;
            if (child_val > highest_val) {
                highest_val = child_val;
                sel = child;
            }
        }
        if (sel->state.sideToMove() == chess::Color::BLACK) {
            topLine += std::to_string(i) + ". ";
            ++i;
        }
        topLine += chess::uci::moveToSan(sel->getParent()->state, sel->move) + " ";
    }
    
    return topLine;
}

// Enqueues a search task for execution by the thread pool. This method is part of the ThreadManager nested class, which manages concurrent search tasks.
void Search::ThreadManager::workerSearch() {
    Node* selection = nullptr;
    auto node = search.rootNode;
    
    ++search.rootNode->visits;

    // For internal nodes, select the best child based on a score and recursively expand it
    float highest_puct = std::numeric_limits<float>::lowest();
    for (const auto& child : node->children) {
        float child_val = child->puct_value();
        if (child_val > highest_puct) {
            highest_puct = child_val;
            selection = child;
        }
    }

    // Ensure a proper selection and avoid bottlenecks
    if (selection == nullptr) {
        search.nn_eval_request(1);
        for (const auto& child : node->children) {
            float child_val = child->puct_value();
            if (child_val > highest_puct) {
                highest_puct = child_val;
                selection = child;
            }
        }
        if (selection == nullptr) {
            selection = node->children.front(); // safe selection set, helps avoid bugs especially in positions with few moves
        }
    }
    selection->virtual_loss = true;
    search.expand(selection); // Recursively expand the selected node
}

// Evaluates the root node with the option to apply Dirichlet noise. This is part of the initialization phase of the search.
void Search::ThreadManager::evaluateRoot(const bool noise) {
    auto eval = search.get_evaluation();

    auto node = eval.second;
    auto policy_tensor = eval.first.first;

    std::vector<float> policy(search.policySize);
    policy_tensor = policy_tensor.to(torch::kFloat32).contiguous();
    std::memcpy(policy.data(), policy_tensor.data_ptr<float>(), search.policySize * sizeof(float));

    auto move_map = policy_map::policy_to_moves(policy, node->state, search.tactic_bonus);
    move_map = noise ? applyDirichletNoise(move_map, root_dirichlet_alpha, root_dirichlet_epsilon) : move_map;
    auto nn_eval = std::make_pair(move_map, -eval.first.second.item<float>());
    auto movelist = search.get_moves(node->state);
    for (const auto &move : movelist) {
        node->expand(move, move_map[move], search.container);
    }
    search.transposition_table.addHash(node->state.hash(), nn_eval);
}

// Evaluates a node using the results from a neural network prediction. This method updates the node's information based on the evaluation.
void Search::ThreadManager::evaluate() {
    // std::cout << "s_eval\n";
    auto eval = search.get_evaluation();

    auto node = eval.second;
    auto policy_tensor = eval.first.first;

    std::vector<float> policy(search.policySize);
    policy_tensor = policy_tensor.to(torch::kFloat32).contiguous();
    std::memcpy(policy.data(), policy_tensor.data_ptr<float>(), search.policySize * sizeof(float));
    auto move_map = policy_map::policy_to_moves(policy, node->state, search.tactic_bonus);
    auto nn_eval = std::make_pair(move_map, -eval.first.second.item<float>());
    auto movelist = search.get_moves(node->state);
    // std::cout << "mid_eval\n";
    for (const auto &move : movelist) {
        node->expand(move, move_map[move], search.container);
    }
    // std::cout << "end_eval\n";
    node->in_nnet = false;
    node->nnet_cv.notify_all();
    node->backpropagate(nn_eval.second, search.container);
    search.transposition_table.addHash(node->state.hash(), nn_eval);
}

// Handles the concurrent evaluation of nodes in the neural network evaluation queue.
void Search::ThreadManager::threadEvaluation() {
    ThreadPool pool(search.num_threads);
    for (unsigned int i = 0; i < search.nn_evaluations.size(); ++i) {
        pool.enqueueTask([this] { this->evaluate(); });
    }
}

// Starts the search process, distributing tasks across a thread pool to explore different moves and positions concurrently.
void Search::ThreadManager::startSearch(const bool dirichelet_noise, bool use_time, std::chrono::duration<int> const& max_time) {
    ThreadPool pool(search.num_threads);
    auto num_sims = search.num_simulations;
    auto sent_searches = 0;
    if (dirichelet_noise) {
        search.expandRoot(search.rootNode, true);
        num_sims--;
    }
    else {
        search.expandRoot(search.rootNode, false);
        num_sims--;
    }
    if (!use_time) {
        while(sent_searches < num_sims) {
            ++sent_searches;
            pool.enqueueTask([this] { this->workerSearch(); });
        }
    }
    else {
        pool.stopAfter(max_time, search.rootNode);
        while(!pool.shouldStop()) {
            if (pool.get_size() < num_sims) {
                ++sent_searches;
                pool.enqueueTask([this] { this->workerSearch(); });
            }
        }
    }
}
