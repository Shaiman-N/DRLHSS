#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <nlohmann/json.hpp>

namespace drl {

/**
 * @brief Structure representing telemetry data from sandbox orchestrators
 * 
 * Contains comprehensive behavioral observations including system calls,
 * file I/O, network activity, process information, and behavioral indicators.
 */
struct TelemetryData {
    // Identification
    std::string sandbox_id;
    std::chrono::system_clock::time_point timestamp;
    
    // System call features
    int syscall_count;
    std::vector<std::string> syscall_types;
    
    // File I/O features
    int file_read_count;
    int file_write_count;
    int file_delete_count;
    std::vector<std::string> accessed_paths;
    
    // Network features
    int network_connections;
    std::vector<std::string> contacted_ips;
    int bytes_sent;
    int bytes_received;
    
    // Process features
    int child_processes;
    float cpu_usage;
    float memory_usage;
    
    // Behavioral indicators
    bool registry_modification;
    bool privilege_escalation_attempt;
    bool code_injection_detected;
    
    // Metadata
    std::string artifact_hash;
    std::string artifact_type;
    
    /**
     * @brief Serialize telemetry data to JSON
     */
    nlohmann::json toJson() const;
    
    /**
     * @brief Deserialize telemetry data from JSON
     */
    static TelemetryData fromJson(const nlohmann::json& j);
    
    /**
     * @brief Check if telemetry data is valid
     */
    bool isValid() const;
};

} // namespace drl
