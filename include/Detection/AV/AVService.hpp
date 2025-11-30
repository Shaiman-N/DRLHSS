/**
 * @file AVService.hpp
 * @brief Background antivirus service for DRLHSS
 * 
 * Monitors file system and provides real-time protection
 * Integrates with DRLHSS DRL, Sandbox, and Database systems
 */

#ifndef DRLHSS_AV_SERVICE_HPP
#define DRLHSS_AV_SERVICE_HPP

#include "ScanEngine.hpp"
#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <vector>

namespace drlhss {
namespace detection {
namespace av {

/**
 * @brief Service configuration
 */
struct ServiceConfig {
    std::string model_directory;
    std::string quarantine_directory;
    std::string log_directory;
    std::string database_path;
    
    // Monitoring
    std::vector<std::string> monitored_directories;
    bool monitor_downloads;
    bool monitor_temp;
    bool monitor_startup;
    
    // Scanning
    bool realtime_protection;
    bool scheduled_scan;
    int scheduled_scan_hour; // 2 AM default
    
    // Performance
    int max_concurrent_scans;
    int scan_queue_size;
    
    // Integration
    bool enable_drl;
    bool enable_sandbox;
    bool auto_sandbox_suspicious;
    
    ServiceConfig() : model_directory("models/onnx/"),
                     quarantine_directory("quarantine/"),
                     log_directory("logs/av/"),
                     database_path("data/drlhss.db"),
                     monitor_downloads(true),
                     monitor_temp(true),
                     monitor_startup(true),
                     realtime_protection(true),
                     scheduled_scan(true),
                     scheduled_scan_hour(2),
                     max_concurrent_scans(4),
                     scan_queue_size(100),
                     enable_drl(true),
                     enable_sandbox(true),
                     auto_sandbox_suspicious(true) {}
};

/**
 * @brief AVService - Background antivirus service for DRLHSS
 * 
 * Provides real-time file system monitoring and scheduled scanning
 * Integrates with DRL for intelligent threat detection
 */
class AVService {
public:
    AVService();
    ~AVService();
    
    /**
     * @brief Initialize service
     * @param config Service configuration
     * @return true if initialized successfully
     */
    bool initialize(const ServiceConfig& config);
    
    /**
     * @brief Start the service
     * @return true if started successfully
     */
    bool start();
    
    /**
     * @brief Stop the service
     */
    void stop();
    
    /**
     * @brief Check if service is running
     */
    bool isRunning() const { return running_; }
    
    /**
     * @brief Manually scan a file
     * @param file_path Path to file
     * @return Analysis result
     */
    AnalysisResult scanFile(const std::string& file_path);
    
    /**
     * @brief Get service statistics
     */
    struct ServiceStatistics {
        uint64_t files_monitored;
        uint64_t files_scanned;
        uint64_t threats_detected;
        uint64_t threats_quarantined;
        uint64_t files_sandboxed;
        uint64_t active_malware_objects;
        uint64_t drl_inferences;
        std::chrono::seconds uptime;
        
        ServiceStatistics() : files_monitored(0), files_scanned(0),
                             threats_detected(0), threats_quarantined(0),
                             files_sandboxed(0), active_malware_objects(0),
                             drl_inferences(0), uptime(0) {}
    };
    
    ServiceStatistics getStatistics() const;
    
private:
    bool initialized_;
    std::atomic<bool> running_;
    ServiceConfig config_;
    
    // Core engine
    std::unique_ptr<ScanEngine> scan_engine_;
    
    // Service threads
    std::unique_ptr<std::thread> monitor_thread_;
    std::unique_ptr<std::thread> scheduler_thread_;
    std::vector<std::unique_ptr<std::thread>> worker_threads_;
    
    // Statistics
    ServiceStatistics stats_;
    std::chrono::time_point<std::chrono::steady_clock> start_time_;
    
    // Thread functions
    void monitoringLoop();
    void schedulerLoop();
    void workerLoop();
    
    // File system monitoring
    void monitorDirectory(const std::string& directory);
    void onFileCreated(const std::string& file_path);
    void onFileModified(const std::string& file_path);
    
    // Malware object lifecycle
    std::unique_ptr<MalwareObject> createMalwareObject(const std::string& file_path);
    void processMalwareObject(std::unique_ptr<MalwareObject> obj);
    void terminateMalwareObject(std::unique_ptr<MalwareObject> obj);
    
    // Scheduled tasks
    void performScheduledScan();
    void performSystemScan();
    
    // Logging
    void logThreat(const AnalysisResult& result);
    void logError(const std::string& message);
};

} // namespace av
} // namespace detection
} // namespace drlhss

#endif // DRLHSS_AV_SERVICE_HPP
