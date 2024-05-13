#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>

class ConfigParser {
public:
    ConfigParser(const std::string& filename) {
        parseConfigFile(filename);
    }

    template <typename T>
    T getValue(const std::string& key, T defaultValue) const {
        auto response = get(key);
        std::cout << key << ": " << response << '\n';
        if (response.empty()) {
            return defaultValue;
        }
        if constexpr (std::is_same_v<T, int>) {
            return std::stoi(response);
        } else if constexpr (std::is_same_v<T, float>) {
            return std::stof(response);
        } else if constexpr (std::is_same_v<T, double>) {
            return std::stod(response);
        }

        return defaultValue;
    }

    void config_params();

private:
    std::map<std::string, std::string> configMap;

    std::string get(const std::string& key) const {
        auto it = configMap.find(key);
        if (it != configMap.end()) {
            return it->second;
        }
        return "";
    }

    void parseConfigFile(const std::string& filename);
};
