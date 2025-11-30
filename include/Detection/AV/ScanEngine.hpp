/**
 * @file ScanEngine.hpp
 * @brief Main scanning orchestrator for DRLHSS Antivirus
 * 
 * Manages MalwareObject lifecycle and coordinates analysis
 * Integrates with DRLHSS DRL, Sandbox, and Database systems
 */

#ifndef DRLHSS_SCAN_ENGINE_HPP
#define DRLHSS_SCAN_ENGINE_HPP

#include "MalwareObject.hpp"
#include "FeatureExtractor.hpp"
#include "BehaviorMonitor.hpp"
#include "InferenceEngine.hpp"
#include <string>
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace drlhss {
namespace detection {
namespace av {

/**
 * @brief Scan configuration
 */
struct ScanConfig {
    bool enable_static;
    bool enable_dynamic;
    bool enable_drl;
    bool auto_sandbox;
    float malicious_threshold;
    float suspicious_threshold;
    int dynamic_monitor_duration;
    std::string quarantine_directory;
    
    ScanConfig() : enable_static(true),
                  enable_dynamic(false),
                  enable_drl(true),
                  auto_sandbox(true),
                  malicious_threshold(70.0f),
                  suspicious_threshold(40.0f),
                  dynamic_monitor_duration(15),
                  quarantine_directory("quarantine/") {}
};

/**
 * @brief ScanEngine - Main antivirus scanning engine for DRLHSS
 */
class ScanEngine {
public:
    ScanEngine();
    ~ScanEngine();
    
    /**
     * @brief Initialize engine with models
     * @param model_dir Directory containing model files
     * @param config Scan configuration
     * @return true if initialized successfully
     */
    bool initialize(const std::string& model_dir, 
                   const ScanConfig& config = ScanConfig());
    
    /**
     * @brief Scan a file
     * @param file_path Path to file
     * @return Analysis result
     */
    AnalysisResult scanFile(const std::string& file_path);
    
    /**
     * @brief Scan a running process
     * @param process_id Process ID
     * @return Analysis result
     */
    AnalysisResult scanProcess(uint32_t process_id);
    
    /**
     * @brief Quarantine a file
     * @param file_path Path to file
     * @return true if quarantined successfully
     */
    bool quarantine(const std::string& file_path);
    
    /**
     * @brief Send to sandbox for analysis
     * @param file_path Path to file
     * @return true if sent successfully
     */
    bool sendToSandbox(const std::string& file_path);
    
    /**
     * @brief Get engine statistics
     */
    struct Statistics {
        uint64_t total_scans;
        uint64_t malicious_detected;
        uint64_t false_positives;
        uint64_t quarantined;
        uint64_t sandboxed;
        uint64_t drl_inferences;
        
        Statistics() : total_scans(0), malicious_detected(0),
                      false_positives(0), quarantined(0),
                      sandboxed(0), drl_inferences(0) {}
    };
    
    Statistics getStatistics() const { return stats_; }
    
    /**
     * @brief Check if engine is ready
     */
    bool isReady() const { return initialized_; }
    
private:
    bool initialized_;
    ScanConfig config_;
    Statistics stats_;
    std::string model_dir_;
    
    // Core components
    std::unique_ptr<FeatureExtractor> feature_extractor_;
    std::unique_ptr<InferenceEngine> inference_engine_;
    
    // Active malware objects
    std::queue<std::unique_ptr<MalwareObject>> active_objects_;
    std::mutex objects_mutex_;
    
    // Helper methods
    AnalysisResult performStaticAnalysis(const std::string& file_path);
    AnalysisResult performDynamicAnalysis(const std::string& file_path);
    AnalysisResult performDRLAnalysis(const std::string& file_path, 
                                     const AnalysisResult& current_result);
    AnalysisResult combineResults(const AnalysisResult& static_result,
                                 const AnalysisResult& dynamic_result,
                                 const AnalysisResult& drl_result);
    
    void cleanupTerminatedObjects();
    bool moveToQuarantine(const std::string& file_path);
    bool notifySandbox(const std::string& file_path);
};

} // namespace av
} // namespace detection
} // namespace drlhss

#endif // DRLHSS_SCAN_ENGINE_HPP
