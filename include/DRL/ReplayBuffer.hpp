#pragma once

#include "DRL/Experience.hpp"
#include <deque>
#include <vector>
#include <mutex>
#include <random>
#include <memory>
#include <fstream>

namespace drl {

/**
 * @brief Thread-safe experience replay buffer for DRL training
 * 
 * Stores experiences with FIFO eviction and supports concurrent
 * access for multi-threaded training scenarios.
 */
class ReplayBuffer {
public:
    /**
     * @brief Constructor
     * @param capacity Maximum number of experiences to store
     */
    explicit ReplayBuffer(size_t capacity = 100000);
    
    /**
     * @brief Add experience to buffer
     * @param experience Experience to add
     */
    void add(const Experience& experience);
    
    /**
     * @brief Add experience with individual components
     */
    void add(const std::vector<float>& state, int action, float reward,
             const std::vector<float>& next_state, bool done);
    
    /**
     * @brief Sample random batch of experiences
     * @param batch_size Number of experiences to sample
     * @return Vector of sampled experiences
     * @throws std::invalid_argument if batch_size > buffer size
     */
    std::vector<Experience> sample(size_t batch_size);
    
    /**
     * @brief Get current buffer size
     */
    size_t size() const;
    
    /**
     * @brief Get buffer capacity
     */
    size_t capacity() const { return capacity_; }
    
    /**
     * @brief Check if buffer has enough samples for training
     * @param min_size Minimum required size
     */
    bool isReady(size_t min_size) const;
    
    /**
     * @brief Clear all experiences
     */
    void clear();
    
    /**
     * @brief Get buffer utilization (0.0 to 1.0)
     */
    double getUtilization() const;
    
    /**
     * @brief Save buffer to file for persistence
     * @param filepath Path to save buffer
     * @return True if successful
     */
    bool saveToFile(const std::string& filepath) const;
    
    /**
     * @brief Load buffer from file
     * @param filepath Path to load buffer from
     * @return True if successful
     */
    bool loadFromFile(const std::string& filepath);

private:
    size_t capacity_;
    std::deque<Experience> buffer_;
    mutable std::mutex buffer_mutex_;
    std::mt19937 rng_;
    
    // Statistics
    mutable size_t total_added_ = 0;
    mutable size_t total_sampled_ = 0;
};

} // namespace drl
