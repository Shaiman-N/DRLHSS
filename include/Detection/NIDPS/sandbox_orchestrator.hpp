#ifndef SANDBOX_ORCHESTRATOR_HPP
#define SANDBOX_ORCHESTRATOR_HPP

#include "sandbox.hpp"
#include "packet_data.hpp"
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <functional>

namespace nidps {

class SandboxOrchestrator {
public:
    using BehaviorCallback = std::function<void(const PacketPtr&, const SandboxBehavior&, SandboxType)>;
    
    SandboxOrchestrator(const std::string& positive_image, const std::string& negative_image);
    ~SandboxOrchestrator();
    
    bool initialize();
    void start();
    void stop();
    
    void submitPacket(const PacketPtr& packet);
    void setBehaviorCallback(BehaviorCallback callback);
    
private:
    void processingLoop();
    void processPacket(const PacketPtr& packet);
    void handlePositiveSandbox(const PacketPtr& packet);
    void handleNegativeSandbox(const PacketPtr& packet);
    
    std::unique_ptr<Sandbox> positive_sandbox_;
    std::unique_ptr<Sandbox> negative_sandbox_;
    
    std::queue<PacketPtr> packet_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    std::atomic<bool> running_;
    std::vector<std::thread> worker_threads_;
    
    BehaviorCallback behavior_callback_;
    std::mutex callback_mutex_;
};

} // namespace nidps

#endif // SANDBOX_ORCHESTRATOR_HPP
