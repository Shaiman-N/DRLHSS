#include "sandbox_orchestrator.hpp"
#include <iostream>

namespace nidps {

SandboxOrchestrator::SandboxOrchestrator(const std::string& positive_image, 
                                         const std::string& negative_image)
    : running_(false) {
    positive_sandbox_ = std::make_unique<Sandbox>(SandboxType::POSITIVE, positive_image);
    negative_sandbox_ = std::make_unique<Sandbox>(SandboxType::NEGATIVE, negative_image);
}

SandboxOrchestrator::~SandboxOrchestrator() {
    stop();
}

bool SandboxOrchestrator::initialize() {
    if (!positive_sandbox_->initialize()) {
        std::cerr << "Failed to initialize positive sandbox" << std::endl;
        return false;
    }
    
    if (!negative_sandbox_->initialize()) {
        std::cerr << "Failed to initialize negative sandbox" << std::endl;
        return false;
    }
    
    std::cout << "Sandbox Orchestrator initialized" << std::endl;
    return true;
}

void SandboxOrchestrator::start() {
    if (running_) {
        return;
    }
    
    running_ = true;
    
    // Start worker threads
    int num_workers = 4;
    for (int i = 0; i < num_workers; ++i) {
        worker_threads_.emplace_back(&SandboxOrchestrator::processingLoop, this);
    }
    
    std::cout << "Sandbox Orchestrator started with " << num_workers << " workers" << std::endl;
}

void SandboxOrchestrator::stop() {
    if (!running_) {
        return;
    }
    
    running_ = false;
    queue_cv_.notify_all();
    
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    worker_threads_.clear();
    std::cout << "Sandbox Orchestrator stopped" << std::endl;
}

void SandboxOrchestrator::submitPacket(const PacketPtr& packet) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        packet_queue_.push(packet);
    }
    queue_cv_.notify_one();
}

void SandboxOrchestrator::setBehaviorCallback(BehaviorCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    behavior_callback_ = callback;
}

void SandboxOrchestrator::processingLoop() {
    while (running_) {
        PacketPtr packet;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { 
                return !packet_queue_.empty() || !running_; 
            });
            
            if (!running_) {
                break;
            }
            
            if (!packet_queue_.empty()) {
                packet = packet_queue_.front();
                packet_queue_.pop();
            }
        }
        
        if (packet) {
            processPacket(packet);
        }
    }
}

void SandboxOrchestrator::processPacket(const PacketPtr& packet) {
    packet->status = PacketStatus::PROCESSING;
    
    // Determine which sandbox to use based on packet status
    if (packet->sandbox_pass_count == 0) {
        // First pass - always goes to positive sandbox
        handlePositiveSandbox(packet);
    } else {
        // Second pass - goes to negative sandbox for false negative check
        handleNegativeSandbox(packet);
    }
}

void SandboxOrchestrator::handlePositiveSandbox(const PacketPtr& packet) {
    std::cout << "Processing packet " << packet->packet_id 
              << " in POSITIVE sandbox" << std::endl;
    
    SandboxBehavior behavior = positive_sandbox_->execute(packet);
    packet->sandbox_pass_count++;
    
    if (behavior.isMalicious()) {
        std::cout << "Malware detected in POSITIVE sandbox (score: " 
                  << behavior.threat_score << ")" << std::endl;
        packet->status = PacketStatus::MALICIOUS;
        
        // Clean the packet
        PacketPtr cleaned = positive_sandbox_->cleanPacket(packet);
        
        // Notify callback with behavior
        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (behavior_callback_) {
                behavior_callback_(cleaned, behavior, SandboxType::POSITIVE);
            }
        }
        
        // Send cleaned packet to negative sandbox for false negative check
        cleaned->false_negative_check = true;
        submitPacket(cleaned);
    } else {
        std::cout << "No malware detected in POSITIVE sandbox" << std::endl;
        packet->status = PacketStatus::CLEAN;
        
        // Send to negative sandbox for false negative check
        packet->false_negative_check = true;
        submitPacket(packet);
    }
}

void SandboxOrchestrator::handleNegativeSandbox(const PacketPtr& packet) {
    std::cout << "Processing packet " << packet->packet_id 
              << " in NEGATIVE sandbox (false negative check)" << std::endl;
    
    SandboxBehavior behavior = negative_sandbox_->execute(packet);
    packet->sandbox_pass_count++;
    
    if (behavior.isMalicious()) {
        std::cout << "False negative detected! Malware found in NEGATIVE sandbox (score: " 
                  << behavior.threat_score << ")" << std::endl;
        packet->status = PacketStatus::MALICIOUS;
        
        // Clean the packet again
        PacketPtr cleaned = negative_sandbox_->cleanPacket(packet);
        cleaned->status = PacketStatus::CLEANED;
        
        // Notify callback
        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (behavior_callback_) {
                behavior_callback_(cleaned, behavior, SandboxType::NEGATIVE);
            }
        }
    } else {
        std::cout << "Packet " << packet->packet_id 
                  << " passed NEGATIVE sandbox - fully clean" << std::endl;
        packet->status = PacketStatus::CLEAN;
        
        // Notify callback with clean status
        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (behavior_callback_) {
                SandboxBehavior clean_behavior;
                behavior_callback_(packet, clean_behavior, SandboxType::NEGATIVE);
            }
        }
    }
}

} // namespace nidps
