#include "sandbox.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <sstream>
#include <iomanip>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

namespace nidps {

Sandbox::Sandbox(SandboxType type, const std::string& image_path)
    : type_(type), image_path_(image_path), initialized_(false) {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999);
    
    std::stringstream ss;
    ss << "sandbox_" << (type == SandboxType::POSITIVE ? "pos" : "neg") 
       << "_" << dis(gen);
    sandbox_id_ = ss.str();
}

Sandbox::~Sandbox() {
    reset();
}

bool Sandbox::initialize() {
    setupIsolatedEnvironment();
    
    monitored_paths_.push_back("/tmp/" + sandbox_id_);
    monitored_paths_.push_back("/var/tmp/" + sandbox_id_);
    
    initialized_ = true;
    std::cout << "Sandbox initialized: " << sandbox_id_ 
              << " (Type: " << (type_ == SandboxType::POSITIVE ? "POSITIVE" : "NEGATIVE") 
              << ")" << std::endl;
    return true;
}

void Sandbox::setupIsolatedEnvironment() {
    // Create isolated namespace for sandbox execution
    // In production, this would use Linux namespaces, cgroups, seccomp
    for (const auto& path : monitored_paths_) {
        std::string cmd = "mkdir -p " + path;
        system(cmd.c_str());
    }
}

SandboxBehavior Sandbox::execute(const PacketPtr& packet) {
    SandboxBehavior behavior;
    
    if (!initialized_) {
        std::cerr << "Sandbox not initialized" << std::endl;
        return behavior;
    }
    
    std::cout << "Executing packet " << packet->packet_id 
              << " in " << (type_ == SandboxType::POSITIVE ? "POSITIVE" : "NEGATIVE") 
              << " sandbox" << std::endl;
    
    // Write packet to sandbox environment
    std::string packet_file = monitored_paths_[0] + "/packet_" + std::to_string(packet->packet_id);
    FILE* fp = fopen(packet_file.c_str(), "wb");
    if (fp) {
        fwrite(packet->raw_data.data(), 1, packet->raw_data.size(), fp);
        fclose(fp);
    }
    
    // Monitor behavior during execution
    monitorBehavior(packet, behavior);
    
    // Analyze different aspects
    analyzeFileSystemActivity(behavior);
    analyzeNetworkActivity(behavior);
    analyzeProcessActivity(behavior);
    
    // Calculate final threat score
    calculateThreatScore(behavior);
    
    // Cleanup
    unlink(packet_file.c_str());
    
    return behavior;
}

void Sandbox::monitorBehavior(const PacketPtr& packet, SandboxBehavior& behavior) {
    // Simulate behavior monitoring
    // In production, this would use ptrace, eBPF, or kernel modules
    
    // Check for suspicious patterns in packet data
    const auto& data = packet->raw_data;
    
    // Detect shellcode patterns
    std::vector<uint8_t> shellcode_patterns = {0x90, 0x90, 0x90}; // NOP sled
    auto it = std::search(data.begin(), data.end(), 
                         shellcode_patterns.begin(), shellcode_patterns.end());
    if (it != data.end()) {
        behavior.memory_injection = true;
        behavior.api_calls.push_back("VirtualAlloc");
        behavior.api_calls.push_back("WriteProcessMemory");
    }
    
    // Detect suspicious port usage
    if (packet->dest_port == 4444 || packet->dest_port == 31337 || 
        packet->dest_port == 1337) {
        behavior.network_activity = true;
        behavior.network_connections.push_back(
            packet->dest_ip + ":" + std::to_string(packet->dest_port));
    }
    
    // Detect encoded payloads
    int high_entropy_bytes = 0;
    for (size_t i = 0; i < data.size() - 1; ++i) {
        if (data[i] > 127 && data[i+1] > 127) {
            high_entropy_bytes++;
        }
    }
    if (high_entropy_bytes > data.size() / 4) {
        behavior.suspicious_api_calls = true;
        behavior.api_calls.push_back("CryptDecrypt");
    }
    
    // Detect SQL injection patterns
    std::string data_str(data.begin(), data.end());
    if (data_str.find("' OR '1'='1") != std::string::npos ||
        data_str.find("UNION SELECT") != std::string::npos ||
        data_str.find("DROP TABLE") != std::string::npos) {
        behavior.suspicious_api_calls = true;
        behavior.api_calls.push_back("SQLExecute");
    }
    
    // Detect XSS patterns
    if (data_str.find("<script>") != std::string::npos ||
        data_str.find("javascript:") != std::string::npos) {
        behavior.suspicious_api_calls = true;
        behavior.api_calls.push_back("eval");
    }
}

void Sandbox::analyzeFileSystemActivity(SandboxBehavior& behavior) {
    // Check for file system modifications
    for (const auto& path : monitored_paths_) {
        std::string cmd = "find " + path + " -type f 2>/dev/null | wc -l";
        FILE* pipe = popen(cmd.c_str(), "r");
        if (pipe) {
            char buffer[128];
            if (fgets(buffer, sizeof(buffer), pipe)) {
                int file_count = atoi(buffer);
                if (file_count > 1) {
                    behavior.file_system_modified = true;
                    behavior.accessed_files.push_back(path);
                }
            }
            pclose(pipe);
        }
    }
}

void Sandbox::analyzeNetworkActivity(SandboxBehavior& behavior) {
    // Monitor network connections
    // In production, use netfilter/iptables to monitor sandbox network activity
    if (!behavior.network_connections.empty()) {
        behavior.network_activity = true;
    }
}

void Sandbox::analyzeProcessActivity(SandboxBehavior& behavior) {
    // Check for process creation
    // In production, monitor fork/exec syscalls
    if (!behavior.api_calls.empty()) {
        for (const auto& api : behavior.api_calls) {
            if (api.find("CreateProcess") != std::string::npos ||
                api.find("fork") != std::string::npos) {
                behavior.process_created = true;
                break;
            }
        }
    }
}

void Sandbox::calculateThreatScore(SandboxBehavior& behavior) {
    int score = 0;
    
    if (behavior.memory_injection) score += 40;
    if (behavior.network_activity) score += 20;
    if (behavior.file_system_modified) score += 15;
    if (behavior.process_created) score += 15;
    if (behavior.suspicious_api_calls) score += 10;
    if (behavior.registry_modified) score += 10;
    
    // Additional scoring based on API calls
    score += std::min(static_cast<int>(behavior.api_calls.size()) * 2, 20);
    
    behavior.threat_score = std::min(score, 100);
    
    std::cout << "Threat score: " << behavior.threat_score << "/100" << std::endl;
}

PacketPtr Sandbox::cleanPacket(const PacketPtr& packet) {
    PacketPtr cleaned = std::make_shared<PacketData>(*packet);
    
    std::cout << "Cleaning malicious packet " << packet->packet_id << std::endl;
    
    // Remove malicious payload
    cleaned->raw_data = removeMaliciousPayload(packet->raw_data);
    cleaned->status = PacketStatus::CLEANED;
    
    return cleaned;
}

std::vector<uint8_t> Sandbox::removeMaliciousPayload(const std::vector<uint8_t>& data) {
    std::vector<uint8_t> cleaned;
    
    // Keep only the header portion (first 54 bytes for IP+TCP/UDP)
    size_t header_size = std::min(data.size(), size_t(54));
    cleaned.assign(data.begin(), data.begin() + header_size);
    
    // Replace payload with safe data
    if (data.size() > header_size) {
        size_t payload_size = data.size() - header_size;
        cleaned.resize(data.size(), 0x00);
    }
    
    return cleaned;
}

void Sandbox::reset() {
    // Clean up sandbox environment
    for (const auto& path : monitored_paths_) {
        std::string cmd = "rm -rf " + path;
        system(cmd.c_str());
    }
}

} // namespace nidps
