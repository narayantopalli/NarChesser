#pragma once

#include <vector>
#include <map>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <iostream>
#include "include/utils/random.hpp"

// Cross-platform terminal clearing function
inline void clearTerminal() {
#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif
}

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
    float maxVal = -std::numeric_limits<float>::infinity();

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
    
    // Calculate the sum of gamma samples for normalization
    float sumOfSamples = 0.0;
    std::vector<float> gammaSamples(probabilities.size());
    for (unsigned int i = 0; i < probabilities.size(); ++i) {
        gammaSamples[i] = randomGamma(alpha, 1.0);
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
    return static_cast<float>(static_cast<int>(1.3*tan(1.57*probability)*100))/100;
}

// fast log: https://www.flipcode.com/archives/Fast_log_Function.shtml
inline float fast_log2 (float val)
{
   int * const    exp_ptr = reinterpret_cast <int *> (&val);
   int            x = *exp_ptr;
   const int      log_2 = ((x >> 23) & 255) - 128;
   x &= ~(255 << 23);
   x += 127 << 23;
   *exp_ptr = x;

   val = ((-1.0f/3) * val + 2) * val - 2.0f/3;   // (1)

   return (val + log_2);
} 

inline float fast_log (const float &val)
{
   return (fast_log2 (val) * 0.69314718f);
}
