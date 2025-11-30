#include "Detection/UnifiedDetectionCoordinator.hpp"
#include "Detection/NIDPS/packet_data.hpp"
#include <iostream>
#include <thread>
#include <chrono>

/**
 * @brief Complete integration example showing NIDPS + DRL + Cross-Platform Sandboxes
 * 
 * This example demonstrates:
 * 1. Unified detection coordinator initialization
 * 2. Network packet processing through NIDPS
 * 3. DRL-based decision making
 * 4. Cross-platform sandbox analysis
 * 5. Database persistence
 * 6. Statistics reporting
 */

void printStats(const detection::UnifiedDetectionCoordinator& coordinator) {
    auto stats = coordinator.getStats();
    
    std::cout << "\n========== UNIFIED DETECTION STATISTICS ==========\n";
    std::cout << "Total Detections:     " << stats.total_detections << "\n";
    std::cout << "Network Detections:   " << stats.network_detections << "\n";
    std::cout << "File Detections:      " << stats.file_detections << "\n";
    std::cout << "Behavior Detections:  " << stats.behavior_detections << "\n";
    std::cout << "Malicious Detected:   " << stats.malicious_detected << "\n";
    std::cout << "False Positives:      " << stats.false_positives << "\n";
    std::cout << "Avg Processing Time:  " << stats.avg_processing_time_ms << " ms\n";
    
    std::cout << "\n--- NIDPS Bridge Stats ---\n";
    std::cout << "Packets Processed:    " << stats.nidps_stats.packets_processed << "\n";
    std::cout << "Packets Blocked:      " << stats.nidps_stats.packets_blocked << "\n";
    std::cout << "Packets Quarantined:  " << stats.nidps_stats.packets_quarantined << "\n";
    std::cout << "Sandbox Executions:   " << stats.nidps_stats.sandbox_executions << "\n";
    
    std::cout << "\n--- DRL Stats ---\n";
    std::cout << "DRL Detections:       " << stats.drl_stats.total_detections << "\n";
    std::cout << "DRL Malicious:        " << stats.drl_stats.malicious_detected << "\n";
    std::cout << "Replay Buffer Size:   " << stats.drl_stats.replay_buffer_size << "\n";
    
    std::cout << "\n--- Database Stats ---\n";
    std::cout << "Telemetry Records:    " << stats.db_stats.telemetry_count << "\n";
    std::cout << "Experience Records:   " << stats.db_stats.experience_count << "\n";
    std::cout << "Attack Patterns:      " << stats.db_stats.pattern_count << "\n";
    std::cout << "DB Size:              " << stats.db_stats.db_size_bytes / 1024 << " KB\n";
    std::cout << "==================================================\n\n";
}

nidps::PacketPtr createTestPacket(uint64_t id, bool malicious = false) {
    auto packet = std::make_shared<nidps::PacketData>();
    
    packet->packet_id = id;
    packet->timestamp = std::chrono::system_clock::now();
    packet->source_ip = "192.168.1." + std::to_string(100 + (id % 50));
    packet->dest_ip = "10.0.0." + std::to_string(1 + (id % 10));
    packet->source_port = 1024 + (id % 60000);
    packet->dest_port = malicious ? 4444 : 80; // Port 4444 often used by malware
    packet->protocol = malicious ? 6 : 17; // TCP for malicious, UDP for clean
    packet->status = malicious ? nidps::PacketStatus::SUSPICIOUS : nidps::PacketStatus::CLEAN;
    
    // Create payload
    if (malicious) {
        // Suspicious payload pattern
        packet->raw_data = {
            0x4d, 0x5a, 0x90, 0x00, // MZ header (executable)
            0x03, 0x00, 0x00, 0x00,
            0x04, 0x00, 0x00, 0x00,
            0xff, 0xff, 0x00, 0x00
        };
        
        // Add more suspicious bytes
        for (int i = 0; i < 100; ++i) {
            packet->raw_data.push_back(0x90); // NOP sled
        }
    } else {
        // Normal HTTP-like payload
        std::string http_payload = "GET / HTTP/1.1\r\nHost: example.com\r\n\r\n";
        packet->raw_data.assign(http_payload.begin(), http_payload.end());
    }
    
    return packet;
}

int main(int argc, char** argv) {
    std::cout << "========================================\n";
    std::cout << "  DRLHSS Integrated System Example\n";
    std::cout << "  NIDPS + DRL + Cross-Platform Sandboxes\n";
    std::cout << "========================================\n\n";
    
    // Configuration
    std::string model_path = "models/onnx/mtl_model.onnx";
    std::string db_path = "drlhss_integrated.db";
    
    // Create unified coordinator
    std::cout << "[Main] Creating unified detection coordinator...\n";
    detection::UnifiedDetectionCoordinator coordinator(model_path, db_path);
    
    // Initialize
    std::cout << "[Main] Initializing all systems...\n";
    if (!coordinator.initialize()) {
        std::cerr << "[Main] Failed to initialize coordinator\n";
        return 1;
    }
    
    // Start processing
    std::cout << "[Main] Starting detection processing...\n";
    coordinator.start();
    
    std::cout << "\n[Main] System ready. Processing test packets...\n\n";
    
    // Process test packets
    std::cout << "--- Processing Clean Packets ---\n";
    for (int i = 0; i < 5; ++i) {
        auto packet = createTestPacket(i, false);
        auto response = coordinator.processNetworkPacket(packet);
        
        std::cout << "Packet " << packet->packet_id 
                  << " - Action: " << response.action
                  << " - Confidence: " << response.confidence
                  << " - Malicious: " << (response.is_malicious ? "YES" : "NO") << "\n";
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << "\n--- Processing Suspicious Packets ---\n";
    for (int i = 100; i < 105; ++i) {
        auto packet = createTestPacket(i, true);
        auto response = coordinator.processNetworkPacket(packet);
        
        std::cout << "Packet " << packet->packet_id 
                  << " - Action: " << response.action
                  << " - Confidence: " << response.confidence
                  << " - Malicious: " << (response.is_malicious ? "YES" : "NO")
                  << " - Attack: " << response.attack_type << "\n";
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Wait for processing to complete
    std::cout << "\n[Main] Waiting for background processing...\n";
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Print statistics
    printStats(coordinator);
    
    // Test file detection
    std::cout << "\n--- Testing File Detection ---\n";
    auto file_response = coordinator.processFile(
        "/tmp/suspicious_file.exe",
        "a1b2c3d4e5f6789012345678901234567890abcd"
    );
    std::cout << "File Detection - Action: " << file_response.action
              << " - Confidence: " << file_response.confidence << "\n";
    
    // Test behavior detection
    std::cout << "\n--- Testing Behavior Detection ---\n";
    drl::TelemetryData behavior_telemetry;
    behavior_telemetry.sandbox_id = "behavior_test_001";
    behavior_telemetry.artifact_hash = "behavior_hash_123";
    behavior_telemetry.file_system_modified = true;
    behavior_telemetry.network_activity_detected = true;
    behavior_telemetry.process_created = true;
    behavior_telemetry.threat_score = 75;
    behavior_telemetry.features = {0.8f, 0.9f, 0.7f, 0.6f, 0.85f, 0.75f, 0.9f, 0.8f};
    
    auto behavior_response = coordinator.processBehavior(behavior_telemetry);
    std::cout << "Behavior Detection - Action: " << behavior_response.action
              << " - Confidence: " << behavior_response.confidence
              << " - Attack: " << behavior_response.attack_type << "\n";
    
    // Final statistics
    std::cout << "\n[Main] Waiting for final processing...\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    printStats(coordinator);
    
    // Export detection events
    std::cout << "[Main] Exporting detection events...\n";
    coordinator.exportDetectionEvents("detection_events.csv", 1000);
    
    // Export experiences for training
    std::cout << "[Main] Exporting experiences for training...\n";
    coordinator.getDRLOrchestrator()->exportExperiences("training_experiences.json", 1000);
    
    // Stop coordinator
    std::cout << "\n[Main] Stopping coordinator...\n";
    coordinator.stop();
    
    std::cout << "\n========================================\n";
    std::cout << "  Integration Example Complete!\n";
    std::cout << "========================================\n";
    
    return 0;
}

