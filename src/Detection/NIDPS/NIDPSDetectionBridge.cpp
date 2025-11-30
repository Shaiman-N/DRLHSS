#include "Detection/NIDPSDetectionBridge.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <openssl/sha.h>

namespace detection {

NIDPSDetectionBridge::NIDPSDetectionBridge(
    std::shared_ptr<drl::DRLOrchestrator> drl_orchestrator,
    const std::string& db_path)
    : drl_orchestrator_(drl_orchestrator), db_path_(db_path) {
}

NIDPSDetectionBridge::~NIDPSDetectionBridge() {
    if (positive_sandbox_) {
        positive_sandbox_->cleanup();
    }
    if (negative_sandbox_) {
        negative_sandbox_->cleanup();
    }
}

bool NIDPSDetectionBridge::initialize() {
    std::cout << "[NIDPSDetectionBridge] Initializing..." << std::endl;
    
    // Create cross-platform sandboxes
    sandbox::SandboxConfig positive_config;
    positive_config.sandbox_id = "nidps_positive";
    positive_config.memory_limit_mb = 512;
    positive_config.cpu_limit_percent = 50;
    positive_config.timeout_seconds = 30;
    positive_config.allow_network = true;
    positive_config.read_only_filesystem = false;
    
    positive_sandbox_ = sandbox::SandboxFactory::createSandbox(sandbox::SandboxType::POSITIVE_FP);
    if (!positive_sandbox_ || !positive_sandbox_->initialize(positive_config)) {
        std::cerr << "[NIDPSDetectionBridge] Failed to initialize positive sandbox" << std::endl;
        return false;
    }
    
    sandbox::SandboxConfig negative_config;
    negative_config.sandbox_id = "nidps_negative";
    negative_config.memory_limit_mb = 512;
    negative_config.cpu_limit_percent = 50;
    negative_config.timeout_seconds = 30;
    negative_config.allow_network = true;
    negative_config.read_only_filesystem = false;
    
    negative_sandbox_ = sandbox::SandboxFactory::createSandbox(sandbox::SandboxType::NEGATIVE_FP);
    if (!negative_sandbox_ || !negative_sandbox_->initialize(negative_config)) {
        std::cerr << "[NIDPSDetectionBridge] Failed to initialize negative sandbox" << std::endl;
        return false;
    }
    
    std::cout << "[NIDPSDetectionBridge] Initialized successfully" << std::endl;
    return true;
}

drl::DRLOrchestrator::DetectionResponse NIDPSDetectionBridge::processPacket(
    const nidps::PacketPtr& packet) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Convert packet to telemetry
    drl::TelemetryData telemetry = convertToTelemetry(packet);
    
    // Get DRL decision
    auto response = drl_orchestrator_->processWithDetails(telemetry);
    
    // Update statistics
    packets_processed_++;
    if (response.action == 1) { // Block
        packets_blocked_++;
    } else if (response.action == 2) { // Quarantine
        packets_quarantined_++;
    }
    
    // If high-risk, analyze in sandbox
    if (response.is_malicious && response.confidence > 0.7f) {
        sandbox::SandboxResult sandbox_result = analyzeSandbox(packet, nidps::SandboxType::POSITIVE);
        
        // Update telemetry with sandbox results
        telemetry = convertToTelemetry(packet, &sandbox_result);
        
        // Re-evaluate with sandbox data
        response = drl_orchestrator_->processWithDetails(telemetry);
        
        // Store experience with sandbox feedback
        float reward = computeReward(packet, response.action, sandbox_result);
        storePacketExperience(packet, response.action, reward);
    }
    
    // Invoke callback if set
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        if (decision_callback_) {
            decision_callback_(packet, response.action, response.confidence);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "[NIDPSDetectionBridge] Processed packet " << packet->packet_id 
              << " - Action: " << response.action 
              << " - Confidence: " << response.confidence 
              << " - Time: " << duration.count() << "ms" << std::endl;
    
    return response;
}

sandbox::SandboxResult NIDPSDetectionBridge::analyzeSandbox(
    const nidps::PacketPtr& packet,
    nidps::SandboxType sandbox_type) {
    
    sandbox_executions_++;
    
    sandbox::SandboxInterface* sandbox = (sandbox_type == nidps::SandboxType::POSITIVE) 
        ? positive_sandbox_.get() 
        : negative_sandbox_.get();
    
    if (!sandbox || !sandbox->isReady()) {
        std::cerr << "[NIDPSDetectionBridge] Sandbox not ready" << std::endl;
        return sandbox::SandboxResult();
    }
    
    // Analyze packet data in sandbox
    sandbox::SandboxResult result = sandbox->analyzePacket(packet->raw_data);
    
    std::cout << "[NIDPSDetectionBridge] Sandbox analysis - Threat score: " 
              << result.threat_score << std::endl;
    
    return result;
}

drl::TelemetryData NIDPSDetectionBridge::convertToTelemetry(
    const nidps::PacketPtr& packet,
    const sandbox::SandboxResult* sandbox_result) {
    
    drl::TelemetryData telemetry;
    
    // Basic packet info
    telemetry.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        packet->timestamp.time_since_epoch()).count();
    telemetry.artifact_hash = computePacketHash(packet);
    telemetry.sandbox_id = "nidps_" + std::to_string(packet->packet_id);
    
    // Extract features
    telemetry.features = extractPacketFeatures(packet);
    
    // Add sandbox results if available
    if (sandbox_result) {
        telemetry.file_system_modified = sandbox_result->file_system_modified;
        telemetry.registry_modified = sandbox_result->registry_modified;
        telemetry.network_activity_detected = sandbox_result->network_activity_detected;
        telemetry.process_created = sandbox_result->process_created;
        telemetry.memory_injection_detected = sandbox_result->memory_injection_detected;
        telemetry.suspicious_api_calls = sandbox_result->suspicious_api_calls;
        telemetry.threat_score = sandbox_result->threat_score;
        telemetry.accessed_files = sandbox_result->accessed_files;
        telemetry.network_connections = sandbox_result->network_connections;
        telemetry.api_calls = sandbox_result->api_calls;
    } else {
        // Use packet status as initial threat indicator
        telemetry.threat_score = (packet->status == nidps::PacketStatus::SUSPICIOUS) ? 30 : 0;
    }
    
    return telemetry;
}

sandbox::SandboxResult NIDPSDetectionBridge::convertSandboxBehavior(
    const nidps::SandboxBehavior& behavior) {
    
    sandbox::SandboxResult result;
    
    result.file_system_modified = behavior.file_system_modified;
    result.registry_modified = behavior.registry_modified;
    result.network_activity_detected = behavior.network_activity;
    result.process_created = behavior.process_created;
    result.memory_injection_detected = behavior.memory_injection;
    result.suspicious_api_calls = behavior.suspicious_api_calls;
    result.threat_score = behavior.threat_score;
    result.accessed_files = behavior.accessed_files;
    result.network_connections = behavior.network_connections;
    result.api_calls = behavior.api_calls;
    result.success = true;
    
    return result;
}

void NIDPSDetectionBridge::storePacketExperience(
    const nidps::PacketPtr& packet,
    int action,
    float reward,
    const nidps::PacketPtr& next_packet) {
    
    drl::TelemetryData current_telemetry = convertToTelemetry(packet);
    
    if (next_packet) {
        drl::TelemetryData next_telemetry = convertToTelemetry(next_packet);
        drl_orchestrator_->storeExperience(current_telemetry, action, reward, next_telemetry, false);
    } else {
        drl::TelemetryData empty_telemetry;
        drl_orchestrator_->storeExperience(current_telemetry, action, reward, empty_telemetry, true);
    }
    
    // Learn attack pattern if malicious
    if (reward > 0.5f && action == 1) { // Blocked malicious packet
        std::string attack_type = "network_intrusion";
        if (packet->protocol == 6) attack_type = "tcp_attack";
        else if (packet->protocol == 17) attack_type = "udp_attack";
        
        drl_orchestrator_->learnAttackPattern(current_telemetry, action, reward, 
                                             attack_type, reward);
    }
}

void NIDPSDetectionBridge::setDecisionCallback(PacketDecisionCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    decision_callback_ = callback;
}

NIDPSDetectionBridge::BridgeStats NIDPSDetectionBridge::getStats() const {
    BridgeStats stats;
    stats.packets_processed = packets_processed_.load();
    stats.packets_blocked = packets_blocked_.load();
    stats.packets_quarantined = packets_quarantined_.load();
    stats.sandbox_executions = sandbox_executions_.load();
    
    // Get DRL stats for avg processing time
    auto drl_stats = drl_orchestrator_->getStats();
    stats.avg_processing_time_ms = drl_stats.avg_inference_time_ms;
    
    return stats;
}

std::vector<float> NIDPSDetectionBridge::extractPacketFeatures(const nidps::PacketPtr& packet) {
    std::vector<float> features;
    
    // Use existing features if available
    if (!packet->features.empty()) {
        return packet->features;
    }
    
    // Otherwise extract basic features
    features.resize(16, 0.0f);
    
    // Feature 0: Packet size (normalized)
    features[0] = std::min(packet->raw_data.size() / 1500.0f, 1.0f);
    
    // Feature 1: Protocol type
    features[1] = packet->protocol / 255.0f;
    
    // Feature 2-3: Port numbers (normalized)
    features[2] = packet->source_port / 65535.0f;
    features[3] = packet->dest_port / 65535.0f;
    
    // Feature 4: Status indicator
    features[4] = (packet->status == nidps::PacketStatus::SUSPICIOUS) ? 1.0f : 0.0f;
    
    // Feature 5: Sandbox pass count
    features[5] = std::min(packet->sandbox_pass_count / 5.0f, 1.0f);
    
    // Features 6-15: Payload byte distribution (first 10 bytes normalized)
    for (size_t i = 0; i < 10 && i < packet->raw_data.size(); ++i) {
        features[6 + i] = packet->raw_data[i] / 255.0f;
    }
    
    return features;
}

std::string NIDPSDetectionBridge::computePacketHash(const nidps::PacketPtr& packet) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(packet->raw_data.data(), packet->raw_data.size(), hash);
    
    std::stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    
    return ss.str();
}

float NIDPSDetectionBridge::computeReward(
    const nidps::PacketPtr& packet,
    int action,
    const sandbox::SandboxResult& result) {
    
    float reward = 0.0f;
    
    // Reward based on threat score and action alignment
    if (result.threat_score > 70) {
        // High threat
        if (action == 1) reward = 1.0f;      // Block - correct
        else if (action == 2) reward = 0.7f; // Quarantine - acceptable
        else reward = -1.0f;                 // Allow - wrong
    } else if (result.threat_score > 40) {
        // Medium threat
        if (action == 2) reward = 1.0f;      // Quarantine - correct
        else if (action == 3) reward = 0.8f; // Deep scan - acceptable
        else if (action == 1) reward = 0.5f; // Block - overly cautious
        else reward = -0.5f;                 // Allow - risky
    } else {
        // Low threat
        if (action == 0) reward = 1.0f;      // Allow - correct
        else if (action == 3) reward = 0.6f; // Deep scan - cautious
        else reward = -0.3f;                 // Block/Quarantine - false positive
    }
    
    return reward;
}

} // namespace detection

