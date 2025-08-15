#include "include/search/constants.hpp"

std::string model_directory = "";
float resign_eval_threshold = 0.0;
float temperature_start = 0.0;
float temperature_end = 0.0;
float root_dirichlet_alpha = 0.0;
float root_dirichlet_epsilon = 0.0;
float cpuct_base = 0.0;
float cpuct_init = 0.0;
float cpuct_factor = 0.0;
int checks_before_move = 0;
int growth_before_check = 0;
int thread_count = 0;
int transposition_table_size = 0;
