#pragma once

#include <random>
#include <memory>

class RandomGenerator {
private:
    static std::unique_ptr<RandomGenerator> instance;
    std::mt19937 generator;
    
public:
    // Singleton pattern - get the global instance
    static RandomGenerator& getInstance() {
        return *instance;
    }
    
    // Constructor needs to be public for std::make_unique
    RandomGenerator() : generator(std::random_device{}()) {}
    
    // Get the underlying generator
    std::mt19937& getGenerator() { return generator; }
    
    // Convenience methods for common distributions
    template<typename T>
    T uniform_real(T min, T max) {
        std::uniform_real_distribution<T> dist(min, max);
        return dist(generator);
    }
    
    template<typename T>
    T uniform_int(T min, T max) {
        std::uniform_int_distribution<T> dist(min, max);
        return dist(generator);
    }
    
    bool bernoulli(float probability) {
        std::bernoulli_distribution dist(probability);
        return dist(generator);
    }
    
    template<typename T>
    T discrete(const std::vector<T>& probabilities) {
        std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
        return static_cast<T>(dist(generator));
    }
    
    float gamma(float alpha, float beta = 1.0) {
        std::gamma_distribution<float> dist(alpha, beta);
        return dist(generator);
    }
    
    // Seed the generator (useful for reproducible results)
    void seed(unsigned int seed_value) {
        generator.seed(seed_value);
    }
    
    // Delete copy constructor and assignment operator
    RandomGenerator(const RandomGenerator&) = delete;
    RandomGenerator& operator=(const RandomGenerator&) = delete;
};

// Initialize the static instance (eager initialization)
inline std::unique_ptr<RandomGenerator> RandomGenerator::instance = std::make_unique<RandomGenerator>();

// Global convenience functions
inline std::mt19937& getRandomGenerator() {
    return RandomGenerator::getInstance().getGenerator();
}

inline bool randomBernoulli(float probability) {
    return RandomGenerator::getInstance().bernoulli(probability);
}

inline float randomGamma(float alpha, float beta = 1.0) {
    return RandomGenerator::getInstance().gamma(alpha, beta);
}

template<typename T>
inline T randomDiscrete(const std::vector<T>& probabilities) {
    return RandomGenerator::getInstance().discrete(probabilities);
}
