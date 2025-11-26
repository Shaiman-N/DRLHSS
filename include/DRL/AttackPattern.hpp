#pragma once

#include <string>
#include <vector>
#include <chrono>

namespace drl {

/**
 * @brief Structure representing a learned attack pattern for database storage
 * 
 * Contains telemetry features, action taken, reward, confidence score,
 * and metadata for pattern retrieval and analysis.
 */
struct AttackPattern {
    std::vector<float> telemetry_features;  // Normalized feature vector
    int action_taken;                        // DRL action that was taken
    float reward;                            // Reward received
    std::string attack_type;                 // Classification of attack
    float confidence_score;                  // Confidence in classification
    std::chrono::system_clock::time_point timestamp;  // When pattern was learned
    std::string sandbox_id;                  // Which sandbox observed this
    std::string artifact_hash;               // Hash of the malicious artifact
    
    /**
     * @brief Default constructor
     */
    AttackPattern() 
        : action_taken(0), reward(0.0f), confidence_score(0.0f),
          timestamp(std::chrono::system_clock::now()) {}
    
    /**
     * @brief Check if pattern is valid
     */
    bool isValid() const {
        return !telemetry_features.empty() && 
               !attack_type.empty() &&
               confidence_score >= 0.0f && confidence_score <= 1.0f;
    }
    
    /**
     * @brief Get timestamp as milliseconds since epoch
     */
    int64_t getTimestampMs() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            timestamp.time_since_epoch()).count();
    }
    
    /**
     * @brief Set timestamp from milliseconds since epoch
     */
    void setTimestampMs(int64_t ms) {
        timestamp = std::chrono::system_clock::time_point(
            std::chrono::milliseconds(ms));
    }
};

} // namespace drl
