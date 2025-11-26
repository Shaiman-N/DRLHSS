#pragma once

#include <vector>

namespace drl {

/**
 * @brief Structure representing a single experience tuple for reinforcement learning
 * 
 * Contains state, action, reward, next state, and done flag for DQN training.
 */
struct Experience {
    std::vector<float> state;       // Current state vector
    int action;                      // Action taken
    float reward;                    // Reward received
    std::vector<float> next_state;  // Resulting state
    bool done;                       // Episode termination flag
    
    /**
     * @brief Default constructor
     */
    Experience() : action(0), reward(0.0f), done(false) {}
    
    /**
     * @brief Parameterized constructor
     */
    Experience(const std::vector<float>& s, int a, float r, 
               const std::vector<float>& ns, bool d)
        : state(s), action(a), reward(r), next_state(ns), done(d) {}
    
    /**
     * @brief Copy constructor
     */
    Experience(const Experience& other) = default;
    
    /**
     * @brief Move constructor
     */
    Experience(Experience&& other) noexcept = default;
    
    /**
     * @brief Copy assignment
     */
    Experience& operator=(const Experience& other) = default;
    
    /**
     * @brief Move assignment
     */
    Experience& operator=(Experience&& other) noexcept = default;
    
    /**
     * @brief Check if experience is valid
     */
    bool isValid() const {
        return !state.empty() && !next_state.empty() && 
               state.size() == next_state.size();
    }
};

} // namespace drl
