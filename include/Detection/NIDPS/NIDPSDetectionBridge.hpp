#pragma once

#include "Detection/NIDPS/packet_data.hpp"
#include "Detection/NIDPS/sandbox.hpp"
#include "DRL/DRLOrchestrator.hpp"
#include "DRL/TelemetryData.hpp"
#include "Sandbox/SandboxInterface.hpp"
#include "Sandbox/SandboxFactory.hpp"
#include <memory>
#include <string>
#include <functional>
#include <atomic>

namespace detection {

/**
 * @brief Bridge between NIDPS and DRL systems
 * 
 * Converts NIDPS packet data to DRL telemetry format,
 * coordinates cross-platform sandboxes, and integrates
 * with DRL decision-making.
 */
class NIDPSDetectionBridge {
public:
    using PacketDecisionCallback = std::function<void(const nidps::PacketPtr&, int action, float confidence)>;
    
    /**
     * @brief Constructor
     * @param drl_orchestrator DRL orchestrator instance
     * @param db_path Database path for telemetry storage
     */
    NIDPSDetectionBridge(std::shared_ptr<drl::DRLOrchestrator> drl_orchestrator,
                         const std::string& db_path);
    
    /**
     * @brief Destructor
     */
    ~NIDPSDetectionBridge();
    
    /**
     * @brief Initialize bridge and sandboxes
     * @return True if successful
     */
    bool initialize();
    
    /**
     * @brief Process packet through DRL and sandboxes
     * @param packet Packet to process
     * @return Detection response with action and confidence
     */
    drl::DRLOrchestrator::DetectionResponse processPacket(const nidps::PacketPtr& packet);
    
    /**
     * @brief Analyze packet in cross-platform sandbox
     * @param packet Packet to analyze
     * @param sandbox_type Type of sandbox (positive/negative)
     * @return Sandbox result with behavioral analysis
     */
    sandbox::SandboxResult analyzeSandbox(const nidps::PacketPtr& packet, 
                                          nidps::SandboxType sandbox_type);
    
    /**
     * @brief Convert NIDPS packet to DRL telemetry
     * @param packet NIDPS packet data
     * @param sandbox_result Optional sandbox result
     * @return DRL telemetry data
     */
    drl::TelemetryData convertToTelemetry(const nidps::PacketPtr& packet,
                                          const sandbox::SandboxResult* sandbox_result = nullptr);
    
    /**
     * @brief Convert NIDPS sandbox behavior to cross-platform result
     * @param behavior NIDPS sandbox behavior
     * @return Cross-platform sandbox result
     */
    sandbox::SandboxResult convertSandboxBehavior(const nidps::SandboxBehavior& behavior);
    
    /**
     * @brief Store packet experience for DRL learning
     * @param packet Packet data
     * @param action Action taken
     * @param reward Reward received
     * @param next_packet Next packet (optional)
     */
    void storePacketExperience(const nidps::PacketPtr& packet, int action, float reward,
                               const nidps::PacketPtr& next_packet = nullptr);
    
    /**
     * @brief Set callback for packet decisions
     * @param callback Callback function
     */
    void setDecisionCallback(PacketDecisionCallback callback);
    
    /**
     * @brief Get bridge statistics
     */
    struct BridgeStats {
        uint64_t packets_processed = 0;
        uint64_t packets_blocked = 0;
        uint64_t packets_quarantined = 0;
        uint64_t sandbox_executions = 0;
        double avg_processing_time_ms = 0.0;
    };
    
    BridgeStats getStats() const;

private:
    // Core components
    std::shared_ptr<drl::DRLOrchestrator> drl_orchestrator_;
    std::unique_ptr<sandbox::SandboxInterface> positive_sandbox_;
    std::unique_ptr<sandbox::SandboxInterface> negative_sandbox_;
    
    // Configuration
    std::string db_path_;
    
    // Callback
    PacketDecisionCallback decision_callback_;
    std::mutex callback_mutex_;
    
    // Statistics
    mutable std::atomic<uint64_t> packets_processed_{0};
    mutable std::atomic<uint64_t> packets_blocked_{0};
    mutable std::atomic<uint64_t> packets_quarantined_{0};
    mutable std::atomic<uint64_t> sandbox_executions_{0};
    
    // Helper methods
    std::vector<float> extractPacketFeatures(const nidps::PacketPtr& packet);
    std::string computePacketHash(const nidps::PacketPtr& packet);
    float computeReward(const nidps::PacketPtr& packet, int action, 
                       const sandbox::SandboxResult& result);
};

} // namespace detection

