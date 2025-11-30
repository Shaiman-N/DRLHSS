/**
 * @file AVDetectionBridge.hpp
 * @brief Bridge between Antivirus and DRLHSS components
 * 
 * Integrates the Antivirus system with DRL framework, database,
 * and cross-platform sandboxes for comprehensive malware detection
 */

#ifndef DRLHSS_AV_DETECTION_BRIDGE_HPP
#define DRLHSS_AV_DETECTION_BRIDGE_HPP

#include "AV/AVService.hpp"
#include "AV/ScanEngine.hpp"
#include "DRL/DRLOrchestrator.hpp"
#include "DB/DatabaseManager.hpp"
#include "Sandbox/SandboxFactory.hpp"
#include <memory>
#include <string>
#include <atomic>
#include <functional>
#include <vector>

namespace drlhss {
namespace detection {

/**
 * @brief Bridge between Antivirus and DRLHSS components
 */
class AVDetectionBridge {
public:
    /**
     * @brief Configuration for AV bridge
     */
    struct AVBridgeConfig {
        std::string ml_model_path;
        std::string drl_model_path;
        std::string database_path;
        std::string quarantine_path;
        std::vector<std::string> scan_directories;
        float malware_threshold;
        bool enable_real_time_monitoring;
        bool enable_sandbox_analysis;
        bool enable_drl_inference;
        bool enable_dynamic_analysis;
        
        AVBridgeConfig() : ml_model_path("models/onnx/antivirus_static_model.onnx"),
                          drl_model_path("models/onnx/dqn_model.onnx"),
                          database_path("data/drlhss.db"),
                          quarantine_path("quarantine/"),
                          malware_threshold(0.7f),
                          enable_real_time_monitoring(true),
                          enable_sandbox_analysis(true),
                          enable_drl_inference(true),
                          enable_dynamic_analysis(false) {
            scan_directories = {"/home", "/tmp", "/var"};
        }
    };
    
    /**
     * @brief File scan result from integrated system
     */
    struct IntegratedScanResult {
        std::string file_path;
        std::string file_hash;
        float av_confidence;
        float ml_confidence;
        int drl_action;
        std::vector<float> drl_q_values;
        float drl_confidence;
        sandbox::SandboxResult sandbox_result;
        bool is_malicious;
        std::string threat_classification;
        std::string recommended_action;
        std::vector<std::string> indicators;
        uint64_t file_size;
        std::string file_type;
        std::chrono::milliseconds scan_duration;
        
        IntegratedScanResult() : av_confidence(0.0f), ml_confidence(0.0f),
                                drl_action(-1), drl_confidence(0.0f),
                                is_malicious(false), file_size(0),
                                scan_duration(0) {}
    };
    
    using ScanCallback = std::function<void(const IntegratedScanResult&)>;
    
    /**
     * @brief Constructor
     * @param config Bridge configuration
     */
    explicit AVDetectionBridge(const AVBridgeConfig& config);
    
    /**
     * @brief Destructor
     */
    ~AVDetectionBridge();
    
    /**
     * @brief Initialize all components
     * @return True if successful
     */
    bool initialize();
    
    /**
     * @brief Start real-time monitoring
     */
    void startMonitoring();
    
    /**
     * @brief Stop real-time monitoring
     */
    void stopMonitoring();
    
    /**
     * @brief Scan a single file
     * @param file_path Path to file to scan
     * @return Integrated scan result
     */
    IntegratedScanResult scanFile(const std::string& file_path);
    
    /**
     * @brief Scan a directory recursively
     * @param directory_path Path to directory to scan
     * @return Vector of scan results
     */
    std::vector<IntegratedScanResult> scanDirectory(const std::string& directory_path);
    
    /**
     * @brief Set scan callback for real-time monitoring
     * @param callback Function to call when file is scanned
     */
    void setScanCallback(ScanCallback callback);
    
    /**
     * @brief Update signature database
     * @return True if successful
     */
    bool updateSignatures();
    
    /**
     * @brief Get system statistics
     */
    struct AVStatistics {
        uint64_t files_scanned;
        uint64_t threats_detected;
        uint64_t files_quarantined;
        uint64_t false_positives;
        uint64_t sandbox_analyses;
        uint64_t drl_inferences;
        double avg_scan_time_ms;
        uint64_t ml_detections;
        
        AVStatistics() : files_scanned(0), threats_detected(0),
                        files_quarantined(0), false_positives(0),
                        sandbox_analyses(0), drl_inferences(0),
                        avg_scan_time_ms(0.0), ml_detections(0) {}
    };
    
    AVStatistics getStatistics() const;

private:
    // File processing pipeline
    void onFileDetected(const std::string& file_path);
    void processWithAV(const std::string& file_path);
    void processWithDRL(const std::string& file_path, float av_confidence);
    void processWithSandbox(const std::string& file_path, int drl_action);
    void finalizeDetection(const IntegratedScanResult& result);
    
    // Helper methods
    drl::TelemetryData convertFileToTelemetry(const std::string& file_path);
    std::string classifyThreat(const IntegratedScanResult& result);
    std::string determineAction(const IntegratedScanResult& result);
    void storeDetectionResult(const IntegratedScanResult& result);
    bool quarantineFile(const std::string& file_path);
    std::string calculateFileHash(const std::string& file_path);
    std::string getFileType(const std::string& file_path);
    uint64_t getFileSize(const std::string& file_path);
    
    AVBridgeConfig config_;
    
    // Core components
    std::unique_ptr<av::AVService> av_service_;
    std::unique_ptr<av::ScanEngine> scan_engine_;
    std::unique_ptr<drl::DRLOrchestrator> drl_orchestrator_;
    std::unique_ptr<db::DatabaseManager> database_;
    std::unique_ptr<sandbox::ISandbox> positive_sandbox_;
    std::unique_ptr<sandbox::ISandbox> negative_sandbox_;
    
    // State
    std::atomic<bool> monitoring_active_;
    ScanCallback scan_callback_;
    
    // Statistics
    mutable std::atomic<uint64_t> files_scanned_{0};
    mutable std::atomic<uint64_t> threats_detected_{0};
    mutable std::atomic<uint64_t> files_quarantined_{0};
    mutable std::atomic<uint64_t> false_positives_{0};
    mutable std::atomic<uint64_t> sandbox_analyses_{0};
    mutable std::atomic<uint64_t> drl_inferences_{0};
    mutable std::atomic<double> total_scan_time_ms_{0.0};
    mutable std::atomic<uint64_t> ml_detections_{0};
};

} // namespace detection
} // namespace drlhss

#endif // DRLHSS_AV_DETECTION_BRIDGE_HPP
