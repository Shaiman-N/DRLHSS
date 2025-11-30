#pragma once

#include "Detection/MD/MalwareDetector.h"
#include "Detection/MD/MalwareObject.h"
#include "Detection/MD/MalwareDetectionService.h"
#include "Detection/MD/MalwareProcessingPipeline.h"
#include "Detection/MD/SandboxOrchestrator.h"
#include "Detection/MD/DRLFramework.h"
#include "Detection/MD/RealTimeMonitor.h"
#include "DRL/DRLOrchestrator.hpp"
#include "DB/DatabaseManager.hpp"
#include "Sandbox/SandboxFactory.hpp"
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <functional>
#include <chrono>

namespace detection {

/**
 * @brief Bridge between Malware Detection system and DRLHSS components
 * 
 * This class integrates the Malware Detection system with the DRL framework,
 * database, and cross-platform sandboxes for comprehensive malware analysis.
 */
class MDDetectionBridge {
public:
    /**
     * @brief Configuration for MD bridge
     */
    struct BridgeConfig {
        std::string malware_model_path;
        std::string malimg_model_path;
        std::string drl_model_path;
        std::string database_path;
        std::string scan_directory;
        float detection_threshold;
        bool enable_realtime_monitoring;
        bool enable_sandbox_analysis;
        bool enable_drl_inference;
        bool enable_image_analysis;
        int max_concurrent_scans;
        
        BridgeConfig() : 
            malware_model_path("models/onnx/malware_dcnn_trained.onnx"),
            malimg_model_path("models/onnx/malimg_finetuned_trained.onnx"),
            drl_model_path("models/onnx/dqn_model.onnx"),
            database_path("data/drlhss.db"),
            scan_directory("/tmp/scan"),
            detection_threshold(0.7f),
            enable_realtime_monitoring(true),
            enable_sandbox_analysis(true),
            enable_drl_inference(true),
            enable_image_analysis(true),
            max_concurrent_scans(4) {}
    };
    
    /**
     * @brief Integrated detection result
     */
    struct IntegratedDetectionResult {
        std::string file_path;
        std::string file_hash;
        bool is_malicious;
        float md_confidence;
        float drl_confidence;
        int drl_action;
        std::vector<float> drl_q_values;
        PipelineResult pipeline_result;
        SandboxAnalysisResult sandbox_result;
        std::string threat_classification;
        std::string malware_family;
        std::string recommended_action;
        std::vector<std::string> attack_patterns;
        std::vector<std::string> behavioral_indicators;
        float overall_threat_score;
        std::chrono::milliseconds scan_duration;
        std::chrono::system_clock::time_point scan_timestamp;
    };
    
    using DetectionCallback = std::function<void(const IntegratedDetectionResult&)>;
    using ThreatCallback = std::function<void(const std::string&, const std::string&)>;
    
    /**
     * @brief Constructor
     * @param config Bridge configuration
     */
    explicit MDDetectionBridge(const BridgeConfig& config);
    
    /**
     * @brief Destructor
     */
    ~MDDetectionBridge();
    
    /**
     * @brief Initialize all components
     * @return True if successful
     */
    bool initialize();
    
    /**
     * @brief Start detection services
     */
    void start();
    
    /**
     * @brief Stop detection services
     */
    void stop();
    
    /**
     * @brief Scan a single file
     * @param file_path Path to file to scan
     * @return Integrated detection result
     */
    IntegratedDetectionResult scanFile(const std::string& file_path);
    
    /**
     * @brief Scan a directory
     * @param directory_path Path to directory to scan
     * @return Vector of detection results
     */
    std::vector<IntegratedDetectionResult> scanDirectory(const std::string& directory_path);
    
    /**
     * @brief Scan data packet
     * @param packet Data packet to scan
     * @return Integrated detection result
     */
    IntegratedDetectionResult scanDataPacket(const std::vector<uint8_t>& packet);
    
    /**
     * @brief Set detection callback
     * @param callback Function to call when detection completes
     */
    void setDetectionCallback(DetectionCallback callback);
    
    /**
     * @brief Set real-time threat callback
     * @param callback Function to call when real-time threat detected
     */
    void setThreatCallback(ThreatCallback callback);
    
    /**
     * @brief Get system statistics
     */
    struct BridgeStatistics {
        uint64_t files_scanned;
        uint64_t malware_detected;
        uint64_t false_positives;
        uint64_t false_negatives;
        uint64_t sandbox_analyses;
        uint64_t drl_inferences;
        uint64_t realtime_detections;
        double avg_scan_time_ms;
        double avg_drl_time_ms;
        double avg_sandbox_time_ms;
        std::chrono::system_clock::time_point start_time;
    };
    
    BridgeStatistics getStatistics() const;
    
    /**
     * @brief Update threat intelligence
     * @param threat_data New threat intelligence data
     */
    void updateThreatIntelligence(const std::vector<std::string>& threat_data);
    
    /**
     * @brief Get real-time monitor statistics
     */
    RealTimeMonitor::MonitorStats getMonitorStats() const;

private:
    // File processing pipeline
    void processFile(const std::string& file_path);
    void processWithMD(const std::string& file_path, IntegratedDetectionResult& result);
    void processWithDRL(IntegratedDetectionResult& result);
    void processWithSandbox(IntegratedDetectionResult& result);
    void processWithPipeline(const std::string& file_path, IntegratedDetectionResult& result);
    void finalizeDetection(const IntegratedDetectionResult& result);
    
    // Helper methods
    drl::TelemetryData convertToTelemetry(const IntegratedDetectionResult& result);
    std::string classifyThreat(const IntegratedDetectionResult& result);
    std::string determineMalwareFamily(const IntegratedDetectionResult& result);
    std::string determineAction(const IntegratedDetectionResult& result);
    float calculateOverallThreatScore(const IntegratedDetectionResult& result);
    void storeDetectionResult(const IntegratedDetectionResult& result);
    void quarantineFile(const std::string& file_path);
    void deleteFile(const std::string& file_path);
    std::string calculateFileHash(const std::string& file_path);
    
    // Real-time monitoring callback
    void onRealtimeThreatDetected(const std::string& location, const std::string& description);
    
    BridgeConfig config_;
    
    // Core MD components
    std::shared_ptr<MalwareDetector> malware_detector_;
    std::shared_ptr<MalwareDetectionService> detection_service_;
    std::shared_ptr<MalwareProcessingPipeline> pipeline_;
    std::shared_ptr<SandboxOrchestrator> md_orchestrator_;
    std::shared_ptr<DRLFramework> md_drl_framework_;
    std::unique_ptr<RealTimeMonitor> realtime_monitor_;
    
    // DRLHSS components
    std::unique_ptr<drl::DRLOrchestrator> drl_orchestrator_;
    std::unique_ptr<db::DatabaseManager> database_;
    std::unique_ptr<sandbox::ISandbox> positive_sandbox_;
    std::unique_ptr<sandbox::ISandbox> negative_sandbox_;
    
    // State
    std::atomic<bool> running_;
    DetectionCallback detection_callback_;
    ThreatCallback threat_callback_;
    
    // Statistics
    mutable std::atomic<uint64_t> files_scanned_{0};
    mutable std::atomic<uint64_t> malware_detected_{0};
    mutable std::atomic<uint64_t> false_positives_{0};
    mutable std::atomic<uint64_t> false_negatives_{0};
    mutable std::atomic<uint64_t> sandbox_analyses_{0};
    mutable std::atomic<uint64_t> drl_inferences_{0};
    mutable std::atomic<uint64_t> realtime_detections_{0};
    mutable std::atomic<double> total_scan_time_ms_{0.0};
    mutable std::atomic<double> total_drl_time_ms_{0.0};
    mutable std::atomic<double> total_sandbox_time_ms_{0.0};
    std::chrono::system_clock::time_point start_time_;
};

} // namespace detection
