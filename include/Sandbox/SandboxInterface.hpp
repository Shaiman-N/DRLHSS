#pragma once

#include <string>
#include <vector>
#include <memory>
#include <chrono>

namespace sandbox {

/**
 * @brief Sandbox execution result
 */
struct SandboxResult {
    bool success;
    int exit_code;
    std::string stdout_output;
    std::string stderr_output;
    std::chrono::milliseconds execution_time;
    
    // Behavioral indicators
    bool file_system_modified;
    bool registry_modified;
    bool network_activity_detected;
    bool process_created;
    bool memory_injection_detected;
    bool suspicious_api_calls;
    
    std::vector<std::string> accessed_files;
    std::vector<std::string> network_connections;
    std::vector<std::string> api_calls;
    
    int threat_score;  // 0-100
    
    SandboxResult() : success(false), exit_code(0), execution_time(0),
                     file_system_modified(false), registry_modified(false),
                     network_activity_detected(false), process_created(false),
                     memory_injection_detected(false), suspicious_api_calls(false),
                     threat_score(0) {}
};

/**
 * @brief Sandbox configuration
 */
struct SandboxConfig {
    std::string sandbox_id;
    std::string base_image_path;
    std::string work_directory;
    
    // Resource limits
    uint64_t memory_limit_mb;
    uint32_t cpu_limit_percent;
    uint32_t timeout_seconds;
    
    // Network settings
    bool allow_network;
    std::vector<std::string> allowed_hosts;
    
    // File system settings
    bool read_only_filesystem;
    std::vector<std::string> allowed_paths;
    
    SandboxConfig() : memory_limit_mb(1024), cpu_limit_percent(50),
                     timeout_seconds(60), allow_network(false),
                     read_only_filesystem(true) {}
};

/**
 * @brief Abstract base class for platform-specific sandbox implementations
 * 
 * This interface defines the contract that all platform-specific
 * sandbox implementations must follow.
 */
class ISandbox {
public:
    virtual ~ISandbox() = default;
    
    /**
     * @brief Initialize the sandbox environment
     * @param config Sandbox configuration
     * @return True if initialization successful
     */
    virtual bool initialize(const SandboxConfig& config) = 0;
    
    /**
     * @brief Execute a file in the sandbox
     * @param file_path Path to file to execute
     * @param args Command line arguments
     * @return Sandbox execution result
     */
    virtual SandboxResult execute(const std::string& file_path,
                                  const std::vector<std::string>& args = {}) = 0;
    
    /**
     * @brief Execute a network packet analysis in sandbox
     * @param packet_data Raw packet data
     * @return Sandbox execution result
     */
    virtual SandboxResult analyzePacket(const std::vector<uint8_t>& packet_data) = 0;
    
    /**
     * @brief Clean up sandbox environment
     */
    virtual void cleanup() = 0;
    
    /**
     * @brief Reset sandbox to initial state
     */
    virtual void reset() = 0;
    
    /**
     * @brief Check if sandbox is ready
     */
    virtual bool isReady() const = 0;
    
    /**
     * @brief Get sandbox ID
     */
    virtual std::string getSandboxId() const = 0;
    
    /**
     * @brief Get platform name
     */
    virtual std::string getPlatform() const = 0;
};

/**
 * @brief Sandbox type enumeration
 */
enum class SandboxType {
    POSITIVE_FP,  // False Positive detection sandbox
    NEGATIVE_FN   // False Negative detection sandbox
};

} // namespace sandbox
