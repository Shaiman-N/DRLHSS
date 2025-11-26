/**
 * @file DRLIntegrationExample.cpp
 * @brief Complete integration example showing how to use the DRL system
 * 
 * This example demonstrates:
 * - Initializing the DRL orchestrator
 * - Processing telemetry data
 * - Making detection decisions
 * - Storing experiences
 * - Learning attack patterns
 * - Exporting data for training
 */

#include "DRL/DRLOrchestrator.hpp"
#include "DRL/TelemetryData.hpp"
#include <iostream>
#include <chrono>
#include <thread>

using namespace drl;

// Helper function to create sample telemetry
TelemetryData createSampleTelemetry(bool malicious = false) {
    TelemetryData telemetry;
    
    telemetry.sandbox_id = "sandbox_001";
    telemetry.timestamp = std::chrono::system_clock::now();
    
    if (malicious) {
        // Malicious behavior indicators
        telemetry.syscall_count = 5000;
        telemetry.file_read_count = 200;
        telemetry.file_write_count = 150;
        telemetry.file_delete_count = 50;
        telemetry.network_connections = 30;
        telemetry.bytes_sent = 5000000;
        telemetry.bytes_received = 2000000;
        telemetry.child_processes = 10;
        telemetry.cpu_usage = 85.0f;
        telemetry.memory_usage = 2048.0f;
        telemetry.registry_modification = true;
        telemetry.privilege_escalation_attempt = true;
        telemetry.code_injection_detected = false;
        telemetry.artifact_hash = "abc123malicious";
        telemetry.artifact_type = "executable";
    } else {
        // Benign behavior
        telemetry.syscall_count = 100;
        telemetry.file_read_count = 10;
        telemetry.file_write_count = 5;
        telemetry.file_delete_count = 0;
        telemetry.network_connections = 2;
        telemetry.bytes_sent = 10000;
        telemetry.bytes_received = 50000;
        telemetry.child_processes = 1;
        telemetry.cpu_usage = 15.0f;
        telemetry.memory_usage = 256.0f;
        telemetry.registry_modification = false;
        telemetry.privilege_escalation_attempt = false;
        telemetry.code_injection_detected = false;
        telemetry.artifact_hash = "def456benign";
        telemetry.artifact_type = "document";
    }
    
    return telemetry;
}

int main(int argc, char** argv) {
    std::cout << "=== DRL Malware Detection System - Integration Example ===" << std::endl;
    
    // Configuration
    std::string model_path = "../../models/onnx/dqn_model.onnx";
    std::string db_path = "../../data/drl_system.db";
    int feature_dim = 16;
    
    // Initialize orchestrator
    std::cout << "\n[1] Initializing DRL Orchestrator..." << std::endl;
    DRLOrchestrator orchestrator(model_path, db_path, feature_dim);
    
    if (!orchestrator.initialize()) {
        std::cerr << "Failed to initialize orchestrator!" << std::endl;
        return 1;
    }
    
    std::cout << "✓ Orchestrator initialized successfully" << std::endl;
    
    // Start background pattern learning
    std::cout << "\n[2] Starting background pattern learning..." << std::endl;
    orchestrator.startPatternLearning();
    std::cout << "✓ Pattern learning started" << std::endl;
    
    // Process benign samples
    std::cout << "\n[3] Processing benign samples..." << std::endl;
    for (int i = 0; i < 10; ++i) {
        TelemetryData telemetry = createSampleTelemetry(false);
        telemetry.sandbox_id = "sandbox_benign_" + std::to_string(i);
        
        auto response = orchestrator.processWithDetails(telemetry);
        
        std::cout << "  Sample " << i << ": Action=" << response.action 
                  << ", Confidence=" << response.confidence
                  << ", Type=" << response.attack_type << std::endl;
        
        // Store experience (reward for correct classification)
        float reward = response.is_malicious ? -1.0f : 1.0f;
        TelemetryData next_telemetry = createSampleTelemetry(false);
        orchestrator.storeExperience(telemetry, response.action, reward, next_telemetry, false);
    }
    
    // Process malicious samples
    std::cout << "\n[4] Processing malicious samples..." << std::endl;
    for (int i = 0; i < 10; ++i) {
        TelemetryData telemetry = createSampleTelemetry(true);
        telemetry.sandbox_id = "sandbox_malicious_" + std::to_string(i);
        
        auto response = orchestrator.processWithDetails(telemetry);
        
        std::cout << "  Sample " << i << ": Action=" << response.action 
                  << ", Confidence=" << response.confidence
                  << ", Type=" << response.attack_type << std::endl;
        
        // Store experience (reward for correct classification)
        float reward = response.is_malicious ? 1.0f : -1.0f;
        TelemetryData next_telemetry = createSampleTelemetry(true);
        orchestrator.storeExperience(telemetry, response.action, reward, next_telemetry, false);
        
        // Learn attack pattern if detected correctly
        if (response.is_malicious && reward > 0) {
            orchestrator.learnAttackPattern(telemetry, response.action, reward,
                                           response.attack_type, response.confidence);
        }
    }
    
    // Get system statistics
    std::cout << "\n[5] System Statistics:" << std::endl;
    auto stats = orchestrator.getStats();
    std::cout << "  Total Detections: " << stats.total_detections << std::endl;
    std::cout << "  Malicious Detected: " << stats.malicious_detected << std::endl;
    std::cout << "  Replay Buffer Size: " << stats.replay_buffer_size << std::endl;
    std::cout << "  Avg Inference Time: " << stats.avg_inference_time_ms << " ms" << std::endl;
    std::cout << "  Database Telemetry Count: " << stats.db_stats.telemetry_count << std::endl;
    std::cout << "  Database Experience Count: " << stats.db_stats.experience_count << std::endl;
    std::cout << "  Database Pattern Count: " << stats.db_stats.pattern_count << std::endl;
    std::cout << "  Database Size: " << stats.db_stats.db_size_bytes / 1024 << " KB" << std::endl;
    
    // Export experiences for training
    std::cout << "\n[6] Exporting experiences for training..." << std::endl;
    std::string export_path = "../../data/exported_experiences.json";
    if (orchestrator.exportExperiences(export_path, 1000)) {
        std::cout << "✓ Experiences exported to: " << export_path << std::endl;
    }
    
    // Simulate model hot-reload
    std::cout << "\n[7] Testing model hot-reload..." << std::endl;
    std::cout << "  (Skipping - would reload from new ONNX file)" << std::endl;
    
    // Stop pattern learning
    std::cout << "\n[8] Stopping pattern learning..." << std::endl;
    orchestrator.stopPatternLearning();
    std::cout << "✓ Pattern learning stopped" << std::endl;
    
    std::cout << "\n=== Integration Example Complete ===" << std::endl;
    std::cout << "\nNext steps:" << std::endl;
    std::cout << "1. Train model using: python python/drl_training/train_complete.py" << std::endl;
    std::cout << "2. Export to ONNX and place in models/onnx/" << std::endl;
    std::cout << "3. Run this integration with real model" << std::endl;
    std::cout << "4. Monitor database for patterns and experiences" << std::endl;
    
    return 0;
}
