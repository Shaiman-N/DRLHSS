#pragma once

#include <string>
#include <map>
#include <vector>
#include <chrono>
#include <memory>

namespace telemetry {

/**
 * @brief Telemetry event types
 */
enum class EventType {
    PROCESS,        // Process creation/termination
    FILE,           // File operations
    REGISTRY,       // Registry/config changes
    NETWORK,        // Network connections
    SYSCALL,        // System calls
    API_CALL,       // Application API calls
    MEMORY,         // Memory operations
    IPC,            // Inter-process communication
    SANDBOX,        // Sandbox execution trace
    STATIC_ANALYSIS,// Static file analysis
    USER_BEHAVIOR,  // User behavior patterns
    UNKNOWN
};

/**
 * @brief Convert event type to string
 */
std::string eventTypeToString(EventType type);

/**
 * @brief Convert string to event type
 */
EventType stringToEventType(const std::string& str);

/**
 * @brief Unified telemetry event structure
 * 
 * This structure represents all types of telemetry events
 * collected from the host system.
 */
struct TelemetryEvent {
    // Core fields
    EventType type;
    std::chrono::system_clock::time_point timestamp;
    int pid;
    std::string process_name;
    std::string process_path;
    
    // Flexible attributes for different event types
    std::map<std::string, std::string> attributes;
    
    // Additional data
    std::vector<uint8_t> raw_data;
    float threat_score;
    bool is_suspicious;
    
    // Constructors
    TelemetryEvent();
    TelemetryEvent(EventType t, int process_id, const std::string& proc_name);
    
    // Serialization
    std::string toJSON() const;
    static TelemetryEvent fromJSON(const std::string& json);
    
    // Helper methods
    void setAttribute(const std::string& key, const std::string& value);
    std::string getAttribute(const std::string& key, const std::string& default_val = "") const;
    bool hasAttribute(const std::string& key) const;
    
    // Specific event creators
    static TelemetryEvent createProcessEvent(int pid, const std::string& name, 
                                             const std::string& path, const std::string& action);
    static TelemetryEvent createFileEvent(int pid, const std::string& proc_name,
                                         const std::string& file_path, const std::string& operation);
    static TelemetryEvent createNetworkEvent(int pid, const std::string& proc_name,
                                            const std::string& dst_ip, int dst_port,
                                            const std::string& protocol);
    static TelemetryEvent createSyscallEvent(int pid, const std::string& proc_name,
                                            const std::string& syscall, const std::string& target);
    static TelemetryEvent createRegistryEvent(int pid, const std::string& proc_name,
                                             const std::string& key, const std::string& operation);
};

} // namespace telemetry
