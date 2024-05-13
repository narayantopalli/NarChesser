#pragma once

#include <vector>
#include <map>
#include <cmath>
#include <limits>
#include <random>
#include <numeric>

template<typename T>
void pop_front(std::vector<T> &v)
{
    if (v.size() > 0) {
        v.erase(v.begin());
    }
}

template<typename T>
std::unordered_map<T, float> Softmax(std::unordered_map<T, float>& map) {
    std::unordered_map<T, float> newMap;
    float total = 0.0;
    float maxVal = std::numeric_limits<float>::lowest();

    // Find the maximum value
    for (const auto& index : map) {
        if (index.second > maxVal) {
            maxVal = index.second;
        }
    }
    for (const auto& index : map) {
        total += std::exp(index.second - maxVal);
    }
    for (const auto& index : map) {
        newMap[index.first] = std::exp(index.second - maxVal) / total;
    }
    return newMap;
}


template<typename T>
std::unordered_map<T, float> applyDirichletNoise(const std::unordered_map<T, float>& probabilities, float alpha, float epsilon) {
    std::unordered_map<T, float> noisyProbabilities; // To hold the noisy probabilities
    std::mt19937 rng(std::random_device{}()); // Random number generator
    
    // Calculate the sum of gamma samples for normalization
    float sumOfSamples = 0.0;
    std::vector<float> gammaSamples(probabilities.size());
    for (unsigned int i = 0; i < probabilities.size(); ++i) {
        std::gamma_distribution<float> distribution(alpha, 1.0);
        gammaSamples[i] = distribution(rng);
        sumOfSamples += gammaSamples[i];
    }

    // Normalize gamma samples and apply noise to probabilities
    auto it = probabilities.begin();
    for (unsigned int i = 0; i < probabilities.size(); ++i, ++it) {
        float normalizedSample = gammaSamples[i] / sumOfSamples;
        // Mix original probability with Dirichlet noise
        noisyProbabilities[it->first] = (1 - epsilon) * it->second + epsilon * normalizedSample;
    }

    return noisyProbabilities;
}

inline float probability_to_centipawn(float probability) {
    return static_cast<float>(static_cast<int>(2.5*atanh(probability)*100))/100;
}
