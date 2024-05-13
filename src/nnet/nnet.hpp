#pragma once

#include <iostream>
#include <vector>
#include <mutex>
#include <torch/script.h>
#include <torch/torch.h>

namespace model {
    extern std::mutex gpu_lock;
    inline std::queue<std::pair<torch::Tensor, torch::Tensor>> evaluate(std::queue<torch::Tensor> inputs_tensor, torch::jit::script::Module module, const torch::Device device) {
        std::lock_guard<std::mutex> guard(gpu_lock); // prevents overflowing gpu on multiple threads
        torch::NoGradGuard no_grad;
        module.eval();

        std::vector<torch::Tensor> inputs_vector;
        inputs_vector.reserve(inputs_tensor.size());

        while (!inputs_tensor.empty()) {
            inputs_vector.push_back(std::move(inputs_tensor.front()));
            inputs_tensor.pop();
        }

        torch::Tensor batch = torch::stack(inputs_vector, 0).to(device);

        // Wrap the batch tensor in an IValue before passing it to the model
        std::vector<torch::jit::IValue> inputs = {batch};

        auto outputs = module.forward(inputs).toTuple()->elements();

        // Assuming the model returns a tuple of two tensors for each input
        torch::Tensor policy_output = outputs.at(0).toTensor().to(torch::kCPU);
        torch::Tensor value_output = outputs.at(1).toTensor().to(torch::kCPU);

        // Prepare the container for the results
        std::queue<std::pair<torch::Tensor, torch::Tensor>> results;

        // Iterate over each item in the batch to pair it with its corresponding outputs
        for (unsigned int i = 0; i < batch.size(0); ++i) {
            auto individual_policy = policy_output[i];
            auto individual_value = value_output[i];

            results.push(std::make_pair(individual_policy, individual_value));
        }
        policy_output.reset();
        value_output.reset();

        return results;
    }
}
