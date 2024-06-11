#pragma once

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <chrono>
#include <future>
#include "constants.hpp"

class ThreadPool {
public:
    ThreadPool(size_t numThreads) : stop(false) {
        for(size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                for(;;) {
                    std::unique_lock<std::mutex> lock(queueMutex);
                    condition.wait(lock, [this] { return stop || !tasks.empty(); });
                    if(stop && tasks.empty()) { return; }
                    
                    auto task = std::move(tasks.front());
                    tasks.pop();
                    lock.unlock();
                    task();
                }
            });
        }
    }

    template<class F>
    void enqueueTask(F&& task) {
        std::unique_lock<std::mutex> lock(queueMutex);
        tasks.emplace(std::forward<F>(task));
        lock.unlock();
        condition.notify_one();
    }
    
    // applies early stopping logic by checking 1st derivative of best move visits advantage
    void stopAfter(std::chrono::duration<int> const& duration, const Node* rootNode) {
        std::thread([this, duration, rootNode]() {
            auto start = std::chrono::high_resolution_clock::now();
            auto end = start + duration;
            int checks = 0;
            int prev_difference = 0;
            int i = 1;
            std::string top_move = "";
            while (std::chrono::high_resolution_clock::now() < end) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                if (rootNode->visits.load() > growth_before_check * i) {
                    ++i;
                    int max_visits = 0;
                    int second_to_max_visits = 0;
                    std::string best_move = "";
                    for (const auto& child : rootNode->children) {
                        auto visits = child->visits.load();
                        if (visits > max_visits) {
                            second_to_max_visits = max_visits;
                            max_visits = visits;
                            best_move = chess::uci::moveToUci(child->move);
                        }
                        else if (visits > second_to_max_visits) {
                            second_to_max_visits = visits;
                        }
                    }
                    if (best_move != top_move) {
                        checks = 0;
                        prev_difference = 0;
                        top_move = best_move;
                    }
                    if (max_visits - second_to_max_visits > prev_difference) {
                        if (checks > checks_before_move) {
                            break;
                        }
                        ++checks;
                    }
                    else {
                        checks = 0;
                    }
                    prev_difference = max_visits - second_to_max_visits;
                }
            }
            terminate();
        }).detach();
    }

    void terminate() {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
        tasks = std::queue<std::function<void()>>();
        lock.unlock();
        condition.notify_all();
    }

    bool shouldStop() const {
        return stop;
    }

    size_t get_size() {
        std::lock_guard<std::mutex> lock(queueMutex);
        return tasks.size();
    }

    ~ThreadPool() {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
        lock.unlock();
        condition.notify_all();
        for(std::thread& worker: workers)
            worker.join();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
};