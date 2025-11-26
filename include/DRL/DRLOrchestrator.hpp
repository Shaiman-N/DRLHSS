#pragma once

#include "DRL/DRLInference.hpp"
#include "DRL/EnvironmentAdapter.hpp"
#include "DRL/ReplayBuffer.hpp"
#include "DRL/TelemetryData.hpp"
#include "DRL/AttackPattern.hpp"
#include "DB/DatabaseManager.hpp"
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <thread>

namespace drl {

/**
 * @brief Production-grade DRL orchestrator for malware detection
 * 
 * Coordinates inference, experience collection, pattern learning,
 * and database persistence for real-time threat detection.
 */
class DRLOrchestrator {
public:
    /**
     * @brief Constructor
     * @param model_path Path to ONNX model
     * @param db_path Path to SQLite database
     * @param feature_dim State vector dimension
     */
    DRLOrchestrator(const std::string& model_path, 
                    const std::string& db_path,
                    int feature_dim = 16);
    
    /**
     * @brief Destructor
     */
    ~DRLOrchestrator();
    
    /**
     * @brief Initialize all components
     * @return True if successful
     */
    bool initialize();
    
    /**
     * @brief Process telemetry and make detection decision
     * @param telemetry Raw telemetry data
     * @return Action index (0=allow, 1=block, 2=quarantine, 3=deep_scan)
     */
    int processAndDecide(const TelemetryData& telemetry);
    
    /**
     * @brief Process telemetry with detailed response
     */
    struct DetectionResponse {
        int action;
        std::vector<float> q_values;
        float confidence;
        std::string attack_type;
        bool is_malicious;
    };
    
    DetectionResponse processWithDetails(const TelemetryData& telemetry);
    
    /**
     * @brief Store experience for future training
     * @param telemetry Current telemetry
     * @param action Action taken
     * @param reward Reward received
     * @param next_telemetry Next telemetry
     * @param done Episode done flag
     */
    void storeExperience(const TelemetryData& telemetry, int action, float reward,
                        const TelemetryData& next_telemetry, bool done);
    
    /**
     * @brief Learn attack pattern from successful detection
     * @param telemetry Telemetry data
     * @param action Action taken
     * @param reward Reward received
     * @param attack_type Type of attack detected
     * @param confidence Confidence score
     */
    void learnAttackPattern(const TelemetryData& telemetry, int action, float reward,
                           const std::string& attack_type, float confidence);
    
    /**
     * @brief Hot-reload model from new ONNX file
     * @param new_model_path Path to new model
     * @return True if successful
     */
    bool reloadModel(const std::string& new_model_path);
    
    /**
     * @brief Get system statistics
     */
    struct SystemStats {
        uint64_t total_detections = 0;
        uint64_t malicious_detected = 0;
        uint64_t false_positives = 0;
        double avg_inference_time_ms = 0.0;
        size_t replay_buffer_size = 0;
        db::DatabaseManager::DatabaseStats db_stats;
    };
    
    SystemStats getStats() const;
    
    /**
     * @brief Export experiences for training
     * @param output_path Path to export file
     * @param limit Maximum number of experiences
     * @return True if successful
     */
    bool exportExperiences(const std::string& output_path, int limit = 10000);
    
    /**
     * @brief Start background pattern learning thread
     */
    void startPatternLearning();
    
    /**
     * @brief Stop background pattern learning thread
     */
    void stopPatternLearning();

private:
    // Core components
    std::unique_ptr<DRLInference> inference_;
    std::unique_ptr<EnvironmentAdapter> adapter_;
    std::unique_ptr<ReplayBuffer> replay_buffer_;
    std::unique_ptr<db::DatabaseManager> db_manager_;
    
    // Configuration
    std::string model_path_;
    std::string db_path_;
    int feature_dim_;
    
    // Statistics
    mutable std::atomic<uint64_t> total_detections_{0};
    mutable std::atomic<uint64_t> malicious_detected_{0};
    mutable std::atomic<uint64_t> false_positives_{0};
    
    // Background learning
    std::atomic<bool> learning_active_{false};
    std::unique_ptr<std::thread> learning_thread_;
    
    // Helper methods
    std::string classifyAttackType(const TelemetryData& telemetry, int action);
    float computeConfidence(const std::vector<float>& q_values);
    void backgroundPatternLearning();
};

} // namespace drl
