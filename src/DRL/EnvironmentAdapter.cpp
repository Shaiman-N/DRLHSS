#include "DRL/EnvironmentAdapter.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace drl {

EnvironmentAdapter::EnvironmentAdapter(int feature_dim) 
    : feature_dim_(feature_dim) {
    initializeDefaults();
    initializeNormalization();
    
    // Define ordered feature names
    feature_names_ = {
        "syscall_count",
        "file_read_count", 
        "file_write_count",
        "file_delete_count",
        "network_connections",
        "bytes_sent",
        "bytes_received", 
        "child_processes",
        "cpu_usage",
        "memory_usage",
        "registry_modification",
        "privilege_escalation_attempt",
        "code_injection_detected",
        "file_io_ratio",
        "network_intensity",
        "process_activity"
    };
}

std::vector<float> EnvironmentAdapter::processTelemetry(const TelemetryData& telemetry) {
    std::vector<float> state(feature_dim_, 0.0f);
    
    // Handle missing fields
    TelemetryData complete_telemetry = handleMissingFields(telemetry);
    
    // Extract basic features
    int idx = 0;
    if (idx < feature_dim_) state[idx++] = normalizeFeature("syscall_count", complete_telemetry.syscall_count);
    if (idx < feature_dim_) state[idx++] = normalizeFeature("file_read_count", complete_telemetry.file_read_count);
    if (idx < feature_dim_) state[idx++] = normalizeFeature("file_write_count", complete_telemetry.file_write_count);
    if (idx < feature_dim_) state[idx++] = normalizeFeature("file_delete_count", complete_telemetry.file_delete_count);
    if (idx < feature_dim_) state[idx++] = normalizeFeature("network_connections", complete_telemetry.network_connections);
    if (idx < feature_dim_) state[idx++] = normalizeFeature("bytes_sent", complete_telemetry.bytes_sent);
    if (idx < feature_dim_) state[idx++] = normalizeFeature("bytes_received", complete_telemetry.bytes_received);
    if (idx < feature_dim_) state[idx++] = normalizeFeature("child_processes", complete_telemetry.child_processes);
    if (idx < feature_dim_) state[idx++] = normalizeFeature("cpu_usage", complete_telemetry.cpu_usage);
    if (idx < feature_dim_) state[idx++] = normalizeFeature("memory_usage", complete_telemetry.memory_usage);
    
    // Boolean features
    if (idx < feature_dim_) state[idx++] = complete_telemetry.registry_modification ? 1.0f : 0.0f;
    if (idx < feature_dim_) state[idx++] = complete_telemetry.privilege_escalation_attempt ? 1.0f : 0.0f;
    if (idx < feature_dim_) state[idx++] = complete_telemetry.code_injection_detected ? 1.0f : 0.0f;
    
    // Derived features
    auto derived = extractDerivedFeatures(complete_telemetry);
    for (size_t i = 0; i < derived.size() && idx < feature_dim_; ++i) {
        state[idx++] = derived[i];
    }
    
    return state;
}

TelemetryData EnvironmentAdapter::handleMissingFields(const TelemetryData& telemetry) {
    TelemetryData complete = telemetry;
    
    // Fill missing numeric fields with defaults
    if (complete.syscall_count < 0) complete.syscall_count = 0;
    if (complete.file_read_count < 0) complete.file_read_count = 0;
    if (complete.file_write_count < 0) complete.file_write_count = 0;
    if (complete.file_delete_count < 0) complete.file_delete_count = 0;
    if (complete.network_connections < 0) complete.network_connections = 0;
    if (complete.bytes_sent < 0) complete.bytes_sent = 0;
    if (complete.bytes_received < 0) complete.bytes_received = 0;
    if (complete.child_processes < 0) complete.child_processes = 0;
    if (complete.cpu_usage < 0.0f || complete.cpu_usage > 100.0f) complete.cpu_usage = 0.0f;
    if (complete.memory_usage < 0.0f) complete.memory_usage = 0.0f;
    
    return complete;
}

void EnvironmentAdapter::updateNormalizationParams(const std::vector<TelemetryData>& telemetry_samples) {
    if (telemetry_samples.empty()) return;
    
    // Compute statistics for each feature
    std::unordered_map<std::string, std::vector<float>> feature_values;
    
    for (const auto& telemetry : telemetry_samples) {
        feature_values["syscall_count"].push_back(telemetry.syscall_count);
        feature_values["file_read_count"].push_back(telemetry.file_read_count);
        feature_values["file_write_count"].push_back(telemetry.file_write_count);
        feature_values["file_delete_count"].push_back(telemetry.file_delete_count);
        feature_values["network_connections"].push_back(telemetry.network_connections);
        feature_values["bytes_sent"].push_back(telemetry.bytes_sent);
        feature_values["bytes_received"].push_back(telemetry.bytes_received);
        feature_values["child_processes"].push_back(telemetry.child_processes);
        feature_values["cpu_usage"].push_back(telemetry.cpu_usage);
        feature_values["memory_usage"].push_back(telemetry.memory_usage);
    }
    
    // Update normalization parameters
    for (const auto& [feature_name, values] : feature_values) {
        if (values.empty()) continue;
        
        auto& params = normalization_params_[feature_name];
        params.min_val = *std::min_element(values.begin(), values.end());
        params.max_val = *std::max_element(values.begin(), values.end());
        
        // Compute mean
        params.mean = std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
        
        // Compute standard deviation
        float variance = 0.0f;
        for (float val : values) {
            variance += (val - params.mean) * (val - params.mean);
        }
        params.std_dev = std::sqrt(variance / values.size());
    }
}

void EnvironmentAdapter::initializeDefaults() {
    default_values_ = {
        {"syscall_count", 0.0f},
        {"file_read_count", 0.0f},
        {"file_write_count", 0.0f},
        {"file_delete_count", 0.0f},
        {"network_connections", 0.0f},
        {"bytes_sent", 0.0f},
        {"bytes_received", 0.0f},
        {"child_processes", 0.0f},
        {"cpu_usage", 0.0f},
        {"memory_usage", 0.0f}
    };
}

void EnvironmentAdapter::initializeNormalization() {
    // Initialize with reasonable defaults based on typical malware behavior
    normalization_params_["syscall_count"] = {0, 10000, 1000, 2000};
    normalization_params_["file_read_count"] = {0, 1000, 50, 100};
    normalization_params_["file_write_count"] = {0, 1000, 50, 100};
    normalization_params_["file_delete_count"] = {0, 100, 5, 10};
    normalization_params_["network_connections"] = {0, 100, 10, 20};
    normalization_params_["bytes_sent"] = {0, 1000000, 10000, 50000};
    normalization_params_["bytes_received"] = {0, 1000000, 10000, 50000};
    normalization_params_["child_processes"] = {0, 50, 2, 5};
    normalization_params_["cpu_usage"] = {0, 100, 30, 20};
    normalization_params_["memory_usage"] = {0, 16000, 500, 1000};
}

float EnvironmentAdapter::normalizeFeature(const std::string& feature_name, float value) {
    auto it = normalization_params_.find(feature_name);
    if (it == normalization_params_.end()) {
        return std::clamp(value, 0.0f, 1.0f);
    }
    
    const auto& params = it->second;
    
    // Min-max normalization
    if (params.max_val > params.min_val) {
        float normalized = (value - params.min_val) / (params.max_val - params.min_val);
        return std::clamp(normalized, 0.0f, 1.0f);
    }
    
    return 0.0f;
}

std::vector<float> EnvironmentAdapter::extractDerivedFeatures(const TelemetryData& telemetry) {
    std::vector<float> derived;
    
    // File I/O ratio
    float total_file_ops = telemetry.file_read_count + telemetry.file_write_count;
    float file_io_ratio = (total_file_ops > 0) ? 
        static_cast<float>(telemetry.file_write_count) / total_file_ops : 0.0f;
    derived.push_back(file_io_ratio);
    
    // Network intensity (normalized to MB)
    float network_intensity = (telemetry.bytes_sent + telemetry.bytes_received) / 1000000.0f;
    derived.push_back(std::clamp(network_intensity, 0.0f, 1.0f));
    
    // Process activity
    float process_activity = std::clamp(telemetry.child_processes / 10.0f, 0.0f, 1.0f);
    derived.push_back(process_activity);
    
    return derived;
}

} // namespace drl
