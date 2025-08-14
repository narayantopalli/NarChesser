#pragma once

#include <cmath>
#include <string>
#include "../utils/functions.hpp"
#include "../chess.hpp"

extern std::string model_directory;

extern float resign_eval_threshold;

extern float temperature_start;
extern float temperature_end;

extern float root_dirichlet_alpha;
extern float root_dirichlet_epsilon;

extern float cpuct_base;
extern float cpuct_init;
extern float cpuct_factor;

inline float cpuct(int visits) {return cpuct_init + cpuct_factor * fast_log((visits + cpuct_base) / cpuct_base);}

extern int checks_before_move;
extern int growth_before_check;
extern int thread_count;
extern int transposition_table_size;
