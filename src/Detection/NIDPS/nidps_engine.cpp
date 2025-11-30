#include "nidps_engine.hpp"
#include <iostream>
#include <csignal>

namespace nidps {

NIDPSEngine::NIDPSEngine(const NIDPSConfig& config)
    : config_(config), running_(false) {}

NIDPSEngine::~NIDPSEngine() {
    stop();
}

bool NIDPSEngine::initialize() {
    std::cout << "Initializing NIDPS Engine..." << std::endl;
    
    // Initialize DRL Framework
    drl_framework_ = std::make_unique<DRLFramework>(config_.model_path);
    if (!drl_framework_->initialize()) {
        std::cerr << "Failed to initialize DRL Framework" << std::endl;
        return false;
    }
    
    // Initialize Database
    database_manager_ = std::make_unique<DatabaseManager>(config_.database_path);
    if (!database_manager_->initialize()) {
        std::cerr << "Failed to initialize Database Manager" << std::endl;
        return false;
    }
    
    // Initialize Sandbox Orchestrator
    sandbox_orchestrator_ = std::make_unique<SandboxOrchestrator>(
        config_.positive_sandbox_image,
        config_.negative_sandbox_image
    );
    if (!sandbox_orchestrator_->initialize()) {
        std::cerr << "Failed to initialize Sandbox Orchestrator" << std::endl;
        return false;
    }
    
    // Set sandbox behavior callback
    sandbox_orchestrator_->setBehaviorCallback(
        [this](const PacketPtr& packet, const SandboxBehavior& behavior, SandboxType type) {
            onSandboxBehavior(packet, behavior, type);
        }
    );
    
    // Initialize Packet Processor
    packet_processor_ = std::make_unique<PacketProcessor>();
    packet_processor_->setOutputCallback(
        [this](const PacketPtr& packet) {
            onPacketOutput(packet);
        }
    );
    
    // Initialize Packet Capture
    packet_capture_ = std::make_unique<PacketCapture>(
        config_.network_interface,
        config_.capture_filter
    );
    if (!packet_capture_->initialize()) {
        std::cerr << "Failed to initialize Packet Capture" << std::endl;
        return false;
    }
    
    std::cout << "NIDPS Engine initialized successfully" << std::endl;
    return true;
}

void NIDPSEngine::start() {
    if (running_) {
        return;
    }
    
    running_ = true;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Starting NIDPS System" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Network Interface: " << config_.network_interface << std::endl;
    std::cout << "Model: " << config_.model_path << std::endl;
    std::cout << "Malware Threshold: " << config_.malware_threshold << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Start sandbox orchestrator
    sandbox_orchestrator_->start();
    
    // Start packet capture
    packet_capture_->start([this](PacketPtr packet) {
        onPacketCaptured(packet);
    });
    
    std::cout << "NIDPS System is running. Press Ctrl+C to stop." << std::endl;
}

void NIDPSEngine::stop() {
    if (!running_) {
        return;
    }
    
    std::cout << "\nStopping NIDPS System..." << std::endl;
    
    running_ = false;
    
    if (packet_capture_) {
        packet_capture_->stop();
    }
    
    if (sandbox_orchestrator_) {
        sandbox_orchestrator_->stop();
    }
    
    printStatistics();
    
    std::cout << "NIDPS System stopped" << std::endl;
}

void NIDPSEngine::onPacketCaptured(PacketPtr packet) {
    if (!running_) {
        return;
    }
    
    std::cout << "\n[CAPTURE] Packet " << packet->packet_id 
              << " captured from " << packet->source_ip << std::endl;
    
    // Initial malware detection using DRL
    processInitialDetection(packet);
}

void NIDPSEngine::processInitialDetection(PacketPtr packet) {
    float confidence = 0.0f;
    bool is_malicious = drl_framework_->detectMalware(packet, confidence);
    
    std::cout << "[DRL] Initial detection - Confidence: " << confidence 
              << " (Threshold: " << config_.malware_threshold << ")" << std::endl;
    
    if (is_malicious && confidence >= config_.malware_threshold) {
        std::cout << "[DRL] Suspicious packet detected - sending to POSITIVE sandbox" 
                  << std::endl;
        packet->status = PacketStatus::SUSPICIOUS;
    } else {
        std::cout << "[DRL] Packet appears clean - sending to NEGATIVE sandbox for verification" 
                  << std::endl;
        packet->status = PacketStatus::CLEAN;
        packet->false_negative_check = true;
    }
    
    // Submit to sandbox orchestrator
    sandbox_orchestrator_->submitPacket(packet);
}

void NIDPSEngine::onSandboxBehavior(const PacketPtr& packet, 
                                    const SandboxBehavior& behavior, 
                                    SandboxType type) {
    std::string sandbox_name = (type == SandboxType::POSITIVE) ? "POSITIVE" : "NEGATIVE";
    
    std::cout << "[" << sandbox_name << " SANDBOX] Packet " << packet->packet_id 
              << " analysis complete" << std::endl;
    
    if (behavior.isMalicious()) {
        handleMaliciousDetection(packet, behavior, type);
    } else {
        handleCleanPacket(packet, type);
    }
}

void NIDPSEngine::handleMaliciousDetection(PacketPtr packet, 
                                           const SandboxBehavior& behavior, 
                                           SandboxType type) {
    std::string sandbox_name = (type == SandboxType::POSITIVE) ? "POSITIVE" : "NEGATIVE";
    
    std::cout << "[" << sandbox_name << " SANDBOX] MALWARE DETECTED!" << std::endl;
    std::cout << "  Threat Score: " << behavior.threat_score << "/100" << std::endl;
    std::cout << "  Memory Injection: " << (behavior.memory_injection ? "YES" : "NO") << std::endl;
    std::cout << "  Network Activity: " << (behavior.network_activity ? "YES" : "NO") << std::endl;
    std::cout << "  File System Modified: " << (behavior.file_system_modified ? "YES" : "NO") << std::endl;
    std::cout << "  Process Created: " << (behavior.process_created ? "YES" : "NO") << std::endl;
    
    // Learn attack pattern
    AttackPattern pattern = drl_framework_->learnFromBehavior(packet, behavior);
    
    // Store in database
    database_manager_->storeAttackPattern(pattern);
    database_manager_->storePacketLog(packet);
    
    // If packet has completed both sandbox passes, send to host
    if (packet->sandbox_pass_count >= 2) {
        packet_processor_->sendToHost(packet);
    }
}

void NIDPSEngine::handleCleanPacket(PacketPtr packet, SandboxType type) {
    std::string sandbox_name = (type == SandboxType::POSITIVE) ? "POSITIVE" : "NEGATIVE";
    
    std::cout << "[" << sandbox_name << " SANDBOX] Packet " << packet->packet_id 
              << " is clean" << std::endl;
    
    // Log packet
    database_manager_->storePacketLog(packet);
    
    // If packet has completed both sandbox passes, send to host
    if (packet->sandbox_pass_count >= 2) {
        packet_processor_->sendToHost(packet);
    }
}

void NIDPSEngine::onPacketOutput(const PacketPtr& packet) {
    std::cout << "[OUTPUT] Packet " << packet->packet_id 
              << " delivered to host system" << std::endl;
}

void NIDPSEngine::printStatistics() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "NIDPS Statistics" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total Processed: " << packet_processor_->getProcessedCount() << std::endl;
    std::cout << "Clean Packets: " << packet_processor_->getCleanCount() << std::endl;
    std::cout << "Malicious Packets: " << packet_processor_->getMaliciousCount() << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Update database statistics
    database_manager_->updateStatistics("total_processed", 
                                       packet_processor_->getProcessedCount());
    database_manager_->updateStatistics("clean_packets", 
                                       packet_processor_->getCleanCount());
    database_manager_->updateStatistics("malicious_packets", 
                                       packet_processor_->getMaliciousCount());
}

} // namespace nidps
