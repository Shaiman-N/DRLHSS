/**
 * @file BehaviorMonitor.hpp
 * @brief Runtime behavior monitoring for DRLHSS
 * 
 * Monitors runtime behavior of processes
 * Extracts 500 API call pattern features for dynamic analysis
 */

#ifndef DRLHSS_BEHAVIOR_MONITOR_HPP
#define DRLHSS_BEHAVIOR_MONITOR_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <chrono>
#include <thread>
#include <atomic>

namespace drlhss {
namespace detection {
namespace av {

/**
 * @brief Behavioral indicators captured during monitoring
 */
struct BehaviorData {
    std::vector<std::string> api_calls;
    std::map<std::string, int> api_frequency;
    
    // System activity
    int network_connections;
    int files_accessed;
    int child_processes;
    int registry_operations;
    int memory_allocations;
    
    // Resource usage
    float avg_cpu_percent;
    float max_cpu_percent;
    float avg_memory_mb;
    float max_memory_mb;
    
    // Timing
    std::chrono::milliseconds monitoring_duration;
    
    BehaviorData() : network_connections(0), files_accessed(0),
                    child_processes(0), registry_operations(0),
                    memory_allocations(0), avg_cpu_percent(0.0f),
                    max_cpu_percent(0.0f), avg_memory_mb(0.0f),
                    max_memory_mb(0.0f), monitoring_duration(0) {}
};

/**
 * @brief BehaviorMonitor - Real-time process behavior monitoring
 */
class BehaviorMonitor {
public:
    /**
     * @brief Constructor
     * @param process_id Process ID to monitor (0 for file execution)
     */
    explicit BehaviorMonitor(uint32_t process_id = 0);
    
    ~BehaviorMonitor();
    
    /**
     * @brief Start monitoring a process
     * @param duration_seconds How long to monitor
     * @return true if monitoring started successfully
     */
    bool startMonitoring(int duration_seconds = 15);
    
    /**
     * @brief Stop monitoring
     */
    void stopMonitoring();
    
    /**
     * @brief Execute file and monitor its behavior
     * @param file_path Path to executable
     * @param duration_seconds Monitoring duration
     * @return true if execution and monitoring succeeded
     */
    bool executeAndMonitor(const std::string& file_path, 
                          int duration_seconds = 15);
    
    /**
     * @brief Extract 500-feature vector for ML model
     * @return Feature vector of size 500
     */
    std::vector<float> extractFeatures();
    
    /**
     * @brief Get raw behavior data
     */
    const BehaviorData& getBehaviorData() const { return behavior_data_; }
    
    /**
     * @brief Check if monitoring is active
     */
    bool isMonitoring() const { return monitoring_active_; }
    
    /**
     * @brief Get last error
     */
    std::string getLastError() const { return last_error_; }
    
private:
    uint32_t process_id_;
    std::atomic<bool> monitoring_active_;
    std::unique_ptr<std::thread> monitor_thread_;
    std::string last_error_;
    
    BehaviorData behavior_data_;
    
    // API vocabulary (loaded from JSON)
    std::map<std::string, int> api_vocabulary_;
    
    // Monitoring methods
    void monitoringLoop(int duration_seconds);
    void captureProcessActivity();
    void captureNetworkActivity();
    void captureFileActivity();
    void captureRegistryActivity();
    void captureMemoryActivity();
    void captureResourceUsage();
    
    // Feature conversion
    std::vector<float> convertToFeatureVector();
    void loadAPIVocabulary(const std::string& vocab_file);
    
    // Helper methods
    bool isProcessRunning(uint32_t pid);
    void mapBehaviorToAPIs();
};

} // namespace av
} // namespace detection
} // namespace drlhss

#endif // DRLHSS_BEHAVIOR_MONITOR_HPP
