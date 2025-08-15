#pragma once

#include <map>
#include <list>
#include <mutex>
#include "include/chess.hpp"

template<typename K, typename V>
class TranspositionTable {
public:
    TranspositionTable() {}
    void addHash(const K key, const V value) {
        std::unique_lock<std::mutex> lock(guard);
        // Check if the key already exists
        auto it = std::find(keys.begin(), keys.end(), key);
        if (it != keys.end()) {
            // Move the key to the end to mark it as recently used
            keys.erase(it);
            keys.push_back(key);
        } else {
            // If adding a new key would exceed the max size, remove the oldest key
            if (table.size() == max_elements) {
                K old_key = keys.front();
                table.erase(old_key);
                keys.pop_front();
            }
            keys.push_back(key);
        }
        
        // Insert or update the key in the map
        table[key] = value;
    }

    V getHash(const K key) {
        // REMOVED FOR PERFORMANCE
        // std::unique_lock<std::mutex> lock(guard);
        // auto it = std::find(keys.begin(), keys.end(), key);
        // if (it != keys.end()) {
        //     // Move the key to the end to mark it as recently used
        //     keys.erase(it);
        //     keys.push_back(key);
        // }
        return table[key];
    }

    bool contains(const K key) const {
        return table.find(key) != table.end();
    }

    inline size_t size() const {
        return table.size();
    }

    void set_size(size_t size) {
        reserved_size = size;
        max_elements = static_cast<size_t>(reserved_size / (sizeof(K) + sizeof(V)));
    }
    size_t max_elements = 0;
private:
    std::unordered_map<K, V> table;
    std::list<K> keys;
    std::mutex guard;
    size_t reserved_size = 0;
};
