#pragma once

#include "Detection/NIDPSDetectionBridge.hpp"
#include "DRL/DRLOrchestrator.hpp"
#include "DB/DatabaseManager.hpp"
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace detection {

/**
 * @brief Unified coordinator for all detection systems
 * 
 * Coordinates NIDPS (network), file detection, and behavior detection
 * with centralized DRL decision-making and database persistence.
 */
class UnifiedDetectionCoordinator {
public:
    /**
     * @brief Detection source type
     */
    enum class DetectionSource {
        NETWORK,    // NIDPS packet detection
        FILE,       // File-based malware detection
        BEHAVIOR    // Behavioral analysis
    };
    
    /**
     * @brief Unified detection event
     */
    struct DetectionEvent {
        DetectionSource source;
        std::string artifact_id;
        std::string artifact_hash;
        drl::TelemetryData telemetry;
        int action;
        float confidence;
        std::string attack_type;
        std::chrono::system_clock::time_point timestamp;
    };
    
    /**
     * @brief Constructor
     * @param model_path Path to DRL ONNX model
     * @param db_path Path to database
     */
    UnifiedDetectionCoordinator(const std::string& model_path,
                                const std::string& db_path);
    
    /**
     * @brief Destructor
     */
    ~UnifiedDetectionCoordinator();
    
    /**
     * @brief Initialize all detection systems
     * @return True if successful
     */
    bool initialize();
    
    /**
     * @brief Start detection processing
     */
    void start();
    
    /**
     * @brief Stop detection processing
     */
    void stop();
    
    /**
     * @brief Process network packet
     * @param packet NIDPS packet
     * @return Detection response
     */
    drl::DRLOrchestrator::DetectionResponse processNetworkPacket(
        const nidps::PacketPtr& packet);
    
    /**
     * @brief Process file for malware detection
     * @param file_path Path to file
     * @param file_hash Hash of file
     * @return Detection response
     */
    drl::DRLOrchestrator::DetectionResponse processFile(
        const std::string& file_path,
        const std::string& file_hash);
    
    /**
     * @brief Process behavioral telemetry
     * @param telemetry Telemetry data
     * @return Detection response
     */
    drl::DRLOrchestrator::DetectionResponse processBehavior(
        const drl::TelemetryData& telemetry);
    
    /**
     * @brief Get NIDPS bridge for direct access
     * @return NIDPS bridge instance
     */
    std::shared_ptr<NIDPSDetectionBridge> getNIDPSBridge() const {
        return nidps_bridge_;
    }
    
    /**
     * @brief Get DRL orchestrator for direct access
     * @return DRL orchestrator instance
     */
    std::shared_ptr<drl::DRLOrchestrator> getDRLOrchestrator() const {
        return drl_orchestrator_;
    }
    
    /**
     * @brief Get database manager for direct access
     * @return Database manager instance
     */
    std::shared_ptr<db::DatabaseManager> getDatabaseManager() const {
        return db_manager_;
    }
    
    /**
     * @brief Get unified statistics
     */
    struct UnifiedStats {
        uint64_t total_detections = 0;
        uint64_t network_detections = 0;
        uint64_t file_detections = 0;
        uint64_t behavior_detections = 0;
        uint64_t malicious_detected = 0;
        uint64_t false_positives = 0;
        double avg_processing_time_ms = 0.0;
        NIDPSDetectionBridge::BridgeStats nidps_stats;
        drl::DRLOrchestrator::SystemStats drl_stats;
        db::DatabaseManager::DatabaseStats db_stats;
    };
    
    UnifiedStats getStats() const;
    
    /**
     * @brief Export detection events to file
     * @param output_path Path to output file
     * @param limit Maximum number of events
     * @return True if successful
     */
    bool exportDetectionEvents(const std::string& output_path, int limit = 10000);
    
    /**
     * @brief Query recent detection events
     * @param source Filter by source (optional)
     * @param limit Maximum number of events
     * @return Vector of detection events
     */
    std::vector<DetectionEvent> queryRecentEvents(
        DetectionSource source = DetectionSource::NETWORK,
        int limit = 100);

private:
    // Core components
    std::shared_ptr<drl::DRLOrchestrator> drl_orchestrator_;
    std::shared_ptr<NIDPSDetectionBridge> nidps_bridge_;
    std::shared_ptr<db::DatabaseManager> db_manager_;
    
    // Configuration
    std::string model_path_;
    std::string db_path_;
    
    // Event processing
    std::queue<DetectionEvent> event_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> running_{false};
    std::unique_ptr<std::thread> processing_thread_;
    
    // Statistics
    mutable std::atomic<uint64_t> total_detections_{0};
    mutable std::atomic<uint64_t> network_detections_{0};
    mutable std::atomic<uint64_t> file_detections_{0};
    mutable std::atomic<uint64_t> behavior_detections_{0};
    mutable std::atomic<uint64_t> malicious_detected_{0};
    
    // Helper methods
    void processingLoop();
    void processEvent(const DetectionEvent& event);
    void storeEvent(const DetectionEvent& event);
    DetectionEvent createEvent(DetectionSource source,
                               const std::string& artifact_id,
                               const drl::TelemetryData& telemetry,
                               const drl::DRLOrchestrator::DetectionResponse& response);
};

} // namespace detection

