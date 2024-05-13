#include "config.hpp"
#include "search/constants.hpp"

void ConfigParser::parseConfigFile(const std::string& filename) {
        std::ifstream configFile(filename);
        if (!configFile.is_open()) {
            std::cerr << "Could not open the file: " << filename << std::endl;
            return;
        }

        std::string line;
        while (std::getline(configFile, line)) {

            line = line.substr(0, line.find('#'));
            line.erase(0, line.find_first_not_of(" \t"));
            line.erase(line.find_last_not_of(" \t") + 1);

            if (line.empty()) continue;

            std::istringstream lineStream(line);
            std::string key, value;

            if (std::getline(lineStream, key, '=') && std::getline(lineStream, value)) {
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);
                configMap[key] = value;
            }
        }
    }


void ConfigParser::config_params() {
    // model_directory
    auto response = get("model_directory");
    if (response == "") {
        model_directory = "../";
    }
    else {
        model_directory = response;
    }
    resign_eval_threshold = getValue("resign_eval_threshold", 0.90f);
    temperature_start = getValue("temperature_start", 1.0f);  
    temperature_end = getValue("temperature_end", 0.10f);
    root_dirichlet_alpha = getValue("root_dirichlet_alpha", 0.3f);
    root_dirichlet_epsilon = getValue("root_dirichlet_epsilon", 0.25f);
    cpuct_base = getValue("cpuct_base", 18368.0f);
    cpuct_init = getValue("cpuct_init", 2.147f);
    cpuct_factor = getValue("cpuct_factor", 2.815f);
    checks_before_move = getValue("checks_before_move", 3);
    growth_before_check = getValue("growth_before_check", 1000);
    thread_count = getValue("thread_count", 4);
    transposition_table_size = getValue("transposition_table_size", 10000000);
}

