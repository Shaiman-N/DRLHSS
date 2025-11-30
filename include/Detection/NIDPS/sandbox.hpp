#ifndef SANDBOX_HPP
#define SANDBOX_HPP

#include "packet_data.hpp"
#include <string>
#include <memory>
#include <vector>

namespace nidps {

enum class SandboxType {
    POSITIVE,  // First sandbox - checks for malware
    NEGATIVE   // Second sandbox - checks for false negatives
};

struct SandboxBehavior {
    bool file_system_modified;
    bool registry_modified;
    bool network_activity;
    bool process_created;
    bool memory_injection;
    bool suspicious_api_calls;
    std::vector<std::string> accessed_files;
    std::vector<std::string> network_connections;
    std::vector<std::string> api_calls;
    int threat_score;
    
    SandboxBehavior() : file_system_modified(false), registry_modified(false),
                       network_activity(false), process_created(false),
                       memory_injection(false), suspicious_api_calls(false),
                       threat_score(0) {}
    
    bool isMalicious() const {
        return threat_score > 50 || memory_injection || 
               (suspicious_api_calls && network_activity);
    }
};

class Sandbox {
public:
    Sandbox(SandboxType type, const std::string& image_path);
    ~Sandbox();
    
    bool initialize();
    SandboxBehavior execute(const PacketPtr& packet);
    PacketPtr cleanPacket(const PacketPtr& packet);
    void reset();
    
    SandboxType getType() const { return type_; }
    std::string getImagePath() const { return image_path_; }
    
private:
    void setupIsolatedEnvironment();
    void monitorBehavior(const PacketPtr& packet, SandboxBehavior& behavior);
    void analyzeFileSystemActivity(SandboxBehavior& behavior);
    void analyzeNetworkActivity(SandboxBehavior& behavior);
    void analyzeProcessActivity(SandboxBehavior& behavior);
    void calculateThreatScore(SandboxBehavior& behavior);
    std::vector<uint8_t> removeMaliciousPayload(const std::vector<uint8_t>& data);
    
    SandboxType type_;
    std::string image_path_;
    std::string sandbox_id_;
    bool initialized_;
    std::vector<std::string> monitored_paths_;
};

} // namespace nidps

#endif // SANDBOX_HPP
