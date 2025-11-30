#ifndef NIDPS_ENGINE_HPP
#define NIDPS_ENGINE_HPP

#include "packet_capture.hpp"
#include "sandbox_orchestrator.hpp"
#include "drl_framework.hpp"
#include "database_manager.hpp"
#include "packet_processor.hpp"
#include <memory>
#include <string>

namespace nidps {

struct NIDPSConfig {
    std::string network_interface;
    std::string capture_filter;
    std::string positive_sandbox_image;
    std::string negative_sandbox_image;
    std::string model_path;
    std::string database_path;
    float malware_threshold;
    
    NIDPSConfig() : network_interface("eth0"), 
                   capture_filter(""),
                   positive_sandbox_image("/var/nidps/sandbox_positive"),
                   negative_sandbox_image("/var/nidps/sandbox_negative"),
                   model_path("mtl_model.onnx"),
                   database_path("nidps.db"),
                   malware_threshold(0.7f) {}
};

class NIDPSEngine {
public:
    NIDPSEngine(const NIDPSConfig& config);
    ~NIDPSEngine();
    
    bool initialize();
    void start();
    void stop();
    
    void printStatistics();
    
private:
    void onPacketCaptured(PacketPtr packet);
    void onSandboxBehavior(const PacketPtr& packet, const SandboxBehavior& behavior, SandboxType type);
    void onPacketOutput(const PacketPtr& packet);
    
    void processInitialDetection(PacketPtr packet);
    void handleMaliciousDetection(PacketPtr packet, const SandboxBehavior& behavior, SandboxType type);
    void handleCleanPacket(PacketPtr packet, SandboxType type);
    
    NIDPSConfig config_;
    
    std::unique_ptr<PacketCapture> packet_capture_;
    std::unique_ptr<SandboxOrchestrator> sandbox_orchestrator_;
    std::unique_ptr<DRLFramework> drl_framework_;
    std::unique_ptr<DatabaseManager> database_manager_;
    std::unique_ptr<PacketProcessor> packet_processor_;
    
    std::atomic<bool> running_;
};

} // namespace nidps

#endif // NIDPS_ENGINE_HPP
