#include "include/model/model.hpp"

namespace model {
    std::mutex device_lock;
    std::queue<std::pair<torch::Tensor, torch::Tensor>> evaluate(std::queue<torch::Tensor> inputs_tensor, torch::jit::script::Module module, const torch::Device device) {
        std::lock_guard<std::mutex> guard(device_lock); // prevents overflowing device on multiple threads
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

        torch::Tensor policy_output = outputs.at(0).toTensor().to(torch::kCPU);
        torch::Tensor value_output = outputs.at(1).toTensor().to(torch::kCPU);

        std::queue<std::pair<torch::Tensor, torch::Tensor>> results;

        for (unsigned int i = 0; i < batch.size(0); ++i) {
            auto individual_policy = policy_output[i];
            auto individual_value = value_output[i];

            results.push(std::make_pair(individual_policy, individual_value));
        }

        // needed to avoid memory leak!
        policy_output.reset();
        value_output.reset();

        return results;
    }
}
