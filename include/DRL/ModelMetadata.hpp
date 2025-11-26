#pragma once

#include <string>
#include <chrono>
#include <nlohmann/json.hpp>

namespace drl {

/**
 * @brief Structure containing metadata about a trained DRL model
 * 
 * Includes training parameters, performance metrics, and versioning information.
 */
struct ModelMetadata {
    // Version information
    std::string model_version;
    std::chrono::system_clock::time_point training_date;
    
    // Training statistics
    int training_episodes;
    float final_average_reward;
    float final_loss;
    
    // Hyperparameters
    float learning_rate;
    float gamma;
    float epsilon_start;
    float epsilon_end;
    int batch_size;
    int target_update_frequency;
    
    // Performance metrics
    float detection_accuracy;
    float false_positive_rate;
    float false_negative_rate;
    
    // Model architecture
    int input_dim;
    int output_dim;
    std::vector<int> hidden_layers;
    
    /**
     * @brief Default constructor
     */
    ModelMetadata()
        : training_episodes(0), final_average_reward(0.0f), final_loss(0.0f),
          learning_rate(0.0001f), gamma(0.99f), epsilon_start(1.0f), epsilon_end(0.1f),
          batch_size(64), target_update_frequency(1000),
          detection_accuracy(0.0f), false_positive_rate(0.0f), false_negative_rate(0.0f),
          input_dim(0), output_dim(0),
          training_date(std::chrono::system_clock::now()) {}
    
    /**
     * @brief Serialize metadata to JSON
     */
    nlohmann::json toJson() const;
    
    /**
     * @brief Deserialize metadata from JSON
     */
    static ModelMetadata fromJson(const nlohmann::json& j);
    
    /**
     * @brief Save metadata to file
     */
    bool saveToFile(const std::string& filepath) const;
    
    /**
     * @brief Load metadata from file
     */
    static ModelMetadata loadFromFile(const std::string& filepath);
};

} // namespace drl
