#include "DRL/TelemetryData.hpp"
#include <stdexcept>

namespace drl {

nlohmann::json TelemetryData::toJson() const {
    nlohmann::json j;
    
    j["sandbox_id"] = sandbox_id;
    j["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
        timestamp.time_since_epoch()).count();
    
    // System call features
    j["syscall_count"] = syscall_count;
    j["syscall_types"] = syscall_types;
    
    // File I/O features
    j["file_read_count"] = file_read_count;
    j["file_write_count"] = file_write_count;
    j["file_delete_count"] = file_delete_count;
    j["accessed_paths"] = accessed_paths;
    
    // Network features
    j["network_connections"] = network_connections;
    j["contacted_ips"] = contacted_ips;
    j["bytes_sent"] = bytes_sent;
    j["bytes_received"] = bytes_received;
    
    // Process features
    j["child_processes"] = child_processes;
    j["cpu_usage"] = cpu_usage;
    j["memory_usage"] = memory_usage;
    
    // Behavioral indicators
    j["registry_modification"] = registry_modification;
    j["privilege_escalation_attempt"] = privilege_escalation_attempt;
    j["code_injection_detected"] = code_injection_detected;
    
    // Metadata
    j["artifact_hash"] = artifact_hash;
    j["artifact_type"] = artifact_type;
    
    return j;
}

TelemetryData TelemetryData::fromJson(const nlohmann::json& j) {
    TelemetryData data;
    
    try {
        data.sandbox_id = j.value("sandbox_id", "");
        
        if (j.contains("timestamp")) {
            auto ms = j["timestamp"].get<int64_t>();
            data.timestamp = std::chrono::system_clock::time_point(
                std::chrono::milliseconds(ms));
        } else {
            data.timestamp = std::chrono::system_clock::now();
        }
        
        // System call features
        data.syscall_count = j.value("syscall_count", 0);
        data.syscall_types = j.value("syscall_types", std::vector<std::string>{});
        
        // File I/O features
        data.file_read_count = j.value("file_read_count", 0);
        data.file_write_count = j.value("file_write_count", 0);
        data.file_delete_count = j.value("file_delete_count", 0);
        data.accessed_paths = j.value("accessed_paths", std::vector<std::string>{});
        
        // Network features
        data.network_connections = j.value("network_connections", 0);
        data.contacted_ips = j.value("contacted_ips", std::vector<std::string>{});
        data.bytes_sent = j.value("bytes_sent", 0);
        data.bytes_received = j.value("bytes_received", 0);
        
        // Process features
        data.child_processes = j.value("child_processes", 0);
        data.cpu_usage = j.value("cpu_usage", 0.0f);
        data.memory_usage = j.value("memory_usage", 0.0f);
        
        // Behavioral indicators
        data.registry_modification = j.value("registry_modification", false);
        data.privilege_escalation_attempt = j.value("privilege_escalation_attempt", false);
        data.code_injection_detected = j.value("code_injection_detected", false);
        
        // Metadata
        data.artifact_hash = j.value("artifact_hash", "");
        data.artifact_type = j.value("artifact_type", "");
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to parse TelemetryData from JSON: " + 
                                 std::string(e.what()));
    }
    
    return data;
}

bool TelemetryData::isValid() const {
    // Basic validation checks
    if (sandbox_id.empty()) return false;
    if (syscall_count < 0) return false;
    if (file_read_count < 0 || file_write_count < 0 || file_delete_count < 0) return false;
    if (network_connections < 0) return false;
    if (bytes_sent < 0 || bytes_received < 0) return false;
    if (child_processes < 0) return false;
    if (cpu_usage < 0.0f || cpu_usage > 100.0f) return false;
    if (memory_usage < 0.0f) return false;
    
    return true;
}

} // namespace drl
