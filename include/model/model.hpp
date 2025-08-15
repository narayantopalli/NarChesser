#pragma once

#include <iostream>
#include <vector>
#include <mutex>
#include <torch/script.h>
#include <torch/torch.h>

namespace model {
    extern std::mutex device_lock;
    extern std::queue<std::pair<torch::Tensor, torch::Tensor>> evaluate(std::queue<torch::Tensor> inputs_tensor, torch::jit::script::Module module, const torch::Device device);
}
