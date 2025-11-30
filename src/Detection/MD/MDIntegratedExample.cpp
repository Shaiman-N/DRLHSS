/**
 * @file MDIntegratedExample.cpp
 * @brief Complete integration example showing Malware Detection + DRL + Sandboxes + Database
 * 
 * This example demonstrates the fully integrated malware detection system with:
 * - File scanning and malware detection (MD)
 * - Deep Reinforcement Learning decisions (DRL)
 * - Cross-platform sandbox analysis (Linux/Windows/macOS)
 * - Database persistence
 * - Real-time threat detection and response
 */

#include "Detection/MDDetectionBridge.hpp"
#include "Sandbox/SandboxFactory.hpp"
#include <iostream>
#include <csignal>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <thread>

std::atomic<bool> g_running(true);

void signalHandler(int signal) {
    std::cout << "\nShutting down gracefully..." << std::endl;
    g_running.store(false);
}

void printBanner() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   DRLHSS - Integrated Malware Detection System              ║" << std::endl;
    std::cout << "║   File Scanning + DRL + Cross-Platform Sandboxes            ║" << std::endl;
    std::cout << "║   Real-Time Monitoring + Attack Pattern Learning            ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;
}

void printSystemInfo() {
    std::cout << "System Information:" << std::endl;
    std::cout << "  Platform: " << sandbox::SandboxFactory::getPlatformName(
        sandbox::SandboxFactory::detectPlatform()) << std::endl;
    
    auto platform = sandbox::SandboxFactory::detectPlatform();
    std::cout << "  Sandbox Support: " 
              << (sandbox::SandboxFactory::isPlatformSupported(platform) ? "✓ Available" : "✗ Not Available")
              << std::endl;
    
    std::cout << std::endl;
}

void printDetectionResult(const detection::MDDetectionBridge::IntegratedDetectionResult& result) {
    std::cout << "\n┌─ Malware Detection Result ────────────────────────────────┐" << std::endl;
    std::cout << "│ File: " << std::filesystem::path(result.file_path).filename().string() << std::endl;
    std::cout << "│ Hash: " << result.file_hash.substr(0, 16) << "..." << std::endl;
    
    if (std::filesystem::exists(result.file_path)) {
        std::cout << "│ Size: " << std::filesystem::file_size(result.file_path) << " bytes" << std::endl;
    }
    
    std::cout << "│" << std::endl;
    std::cout << "│ MD Confidence: " << (result.md_confidence * 100) << "%" << std::endl;
    std::cout << "│ DRL Action: " << result.drl_action 
              << " (0=Allow, 1=Block, 2=Quarantine, 3=DeepScan)" << std::endl;
    std::cout << "│ DRL Confidence: " << (result.drl_confidence * 100) << "%" << std::endl;
    std::cout << "│" << std::endl;
    std::cout << "│ Malicious: " << (result.is_malicious ? "YES ⚠️" : "NO ✓") << std::endl;
    std::cout << "│ Threat Type: " << result.threat_classification << std::endl;
    std::cout << "│ Malware Family: " << result.malware_family << std::endl;
    std::cout << "│ Overall Threat Score: " << (result.overall_threat_score * 100) << "%" << std::endl;
    std::cout << "│ Action: " << result.recommended_action << std::endl;
    std::cout << "│ Scan Duration: " << result.scan_duration.count() << " ms" << std::endl;
    
    if (!result.attack_patterns.empty()) {
        std::cout << "│" << std::endl;
        std::cout << "│ Attack Patterns:" << std::endl;
        for (const auto& pattern : result.attack_patterns) {
            std::cout << "│   - " << pattern << std::endl;
        }
    }
    
    if (!result.behavioral_indicators.empty()) {
        std::cout << "│" << std::endl;
        std::cout << "│ Behavioral Indicators:" << std::endl;
        for (const auto& indicator : result.behavioral_indicators) {
            std::cout << "│   - " << indicator << std::endl;
        }
    }
    
    if (result.sandbox_result.success) {
        std::cout << "│" << std::endl;
        std::cout << "│ Sandbox Analysis:" << std::endl;
        std::cout << "│   Threat Score: " << result.sandbox_result.threat_score << "/100" << std::endl;
        std::cout << "│   File Modified: " << (result.sandbox_result.file_system_modified ? "Yes" : "No") << std::endl;
        std::cout << "│   Registry Modified: " << (result.sandbox_result.registry_modified ? "Yes" : "No") << std::endl;
        std::cout << "│   Network Activity: " << (result.sandbox_result.network_activity_detected ? "Yes" : "No") << std::endl;
        std::cout << "│   Process Created: " << (result.sandbox_result.process_created ? "Yes" : "No") << std::endl;
        std::cout << "│   Execution Time: " << result.sandbox_result.execution_time.count() << "ms" << std::endl;
        
        if (!result.sandbox_result.accessed_files.empty()) {
            std::cout << "│   Files Accessed: " << result.sandbox_result.accessed_files.size() << std::endl;
        }
    }
    
    std::cout << "└────────────────────────────────────────────────────────────┘" << std::endl;
}

void createTestFiles() {
    std::cout << "Creating test files for demonstration..." << std::endl;
    
    // Create test directory
    std::filesystem::create_directories("test_files");
    
    // Create benign test file
    std::ofstream benign("test_files/benign.txt");
    benign << "This is a benign text file for testing." << std::endl;
    benign.close();
    
    // Create suspicious test file (simulated)
    std::ofstream suspicious("test_files/suspicious.exe");
    suspicious << "MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xFF\xFF\x00\x00"; // PE header
    suspicious << "Suspicious content that might trigger detection" << std::endl;
    suspicious.close();
    
    // Create another test file
    std::ofstream script("test_files/script.bat");
    script << "@echo off\n";
    script << "echo This is a test batch script\n";
    script << "pause\n";
    script.close();
    
    // Create a Python script
    std::ofstream python("test_files/test_script.py");
    python << "#!/usr/bin/env python3\n";
    python << "print('Hello from Python')\n";
    python.close();
    
    std::cout << "✓ Test files created in test_files/ directory" << std::endl;
}

void demonstrateRealtimeMonitoring(detection::MDDetectionBridge& bridge) {
    std::cout << "\n═══════════════════════════════════════════════════════════" << std::endl;
    std::cout << "  Real-Time Monitoring Demonstration" << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════" << std::endl;
    
    // Set threat callback
    bridge.setThreatCallback([](const std::string& location, const std::string& description) {
        std::cout << "\n🚨 REAL-TIME THREAT ALERT!" << std::endl;
        std::cout << "   Location: " << location << std::endl;
        std::cout << "   Description: " << description << std::endl;
    });
    
    std::cout << "\nReal-time monitoring is active..." << std::endl;
    std::cout << "Monitoring for:" << std::endl;
    std::cout << "  - File system changes" << std::endl;
    std::cout << "  - Registry modifications" << std::endl;
    std::cout << "  - Startup folder changes" << std::endl;
    std::cout << "  - Network activity" << std::endl;
    
    // Get monitor stats
    auto monitor_stats = bridge.getMonitorStats();
    std::cout << "\nMonitor Statistics:" << std::endl;
    std::cout << "  Files Scanned: " << monitor_stats.filesScanned << std::endl;
    std::cout << "  Threats Detected: " << monitor_stats.threatsDetected << std::endl;
    std::cout << "  Registry Changes: " << monitor_stats.registryChanges << std::endl;
    std::cout << "  Startup Changes: " << monitor_stats.startupChanges << std::endl;
}

int main(int argc, char** argv) {
    // Setup signal handler
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    
    printBanner();
    printSystemInfo();
    
    // Parse command line arguments
    std::string scan_path = "test_files";
    bool create_test = true;
    bool enable_realtime = false;
    
    if (argc > 1) {
        scan_path = argv[1];
        create_test = false;
    }
    
    if (argc > 2 && std::string(argv[2]) == "--realtime") {
        enable_realtime = true;
    }
    
    if (create_test) {
        createTestFiles();
    }
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Scan Path: " << scan_path << std::endl;
    std::cout << "  Malware Model: models/onnx/malware_dcnn_trained.onnx" << std::endl;
    std::cout << "  MalImg Model: models/onnx/malimg_finetuned_trained.onnx" << std::endl;
    std::cout << "  DRL Model: models/onnx/dqn_model.onnx" << std::endl;
    std::cout << "  Database: data/drlhss.db" << std::endl;
    std::cout << "  Real-Time Monitoring: " << (enable_realtime ? "Enabled" : "Disabled") << std::endl;
    std::cout << std::endl;
    
    // Configure MD bridge
    detection::MDDetectionBridge::BridgeConfig config;
    config.malware_model_path = "models/onnx/malware_dcnn_trained.onnx";
    config.malimg_model_path = "models/onnx/malimg_finetuned_trained.onnx";
    config.drl_model_path = "models/onnx/dqn_model.onnx";
    config.database_path = "data/drlhss.db";
    config.scan_directory = scan_path;
    config.detection_threshold = 0.7f;
    config.enable_realtime_monitoring = enable_realtime;
    config.enable_sandbox_analysis = true;
    config.enable_drl_inference = true;
    config.enable_image_analysis = true;
    config.max_concurrent_scans = 4;
    
    // Create and initialize bridge
    std::cout << "[1/3] Initializing Malware Detection Bridge..." << std::endl;
    detection::MDDetectionBridge bridge(config);
    
    if (!bridge.initialize()) {
        std::cerr << "✗ Failed to initialize MD bridge!" << std::endl;
        return 1;
    }
    std::cout << "✓ MD bridge initialized successfully" << std::endl;
    
    // Set detection callback
    std::cout << "[2/3] Setting up detection callback..." << std::endl;
    bridge.setDetectionCallback(printDetectionResult);
    std::cout << "✓ Detection callback configured" << std::endl;
    
    // Start detection
    std::cout << "[3/3] Starting malware detection system..." << std::endl;
    bridge.start();
    std::cout << "✓ Malware detection system started" << std::endl;
    std::cout << std::endl;
    
    // Demonstrate real-time monitoring if enabled
    if (enable_realtime) {
        demonstrateRealtimeMonitoring(bridge);
        std::cout << "\nPress Ctrl+C to stop monitoring and continue to file scanning..." << std::endl;
        
        // Wait for user interrupt or timeout
        int wait_seconds = 10;
        for (int i = 0; i < wait_seconds && g_running.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    
    std::cout << "\n═══════════════════════════════════════════════════════════" << std::endl;
    std::cout << "  Starting file scan..." << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════" << std::endl;
    std::cout << std::endl;
    
    // Scan files
    if (std::filesystem::is_directory(scan_path)) {
        std::cout << "Scanning directory: " << scan_path << std::endl;
        auto results = bridge.scanDirectory(scan_path);
        std::cout << "\nScanned " << results.size() << " files" << std::endl;
    } else if (std::filesystem::is_regular_file(scan_path)) {
        std::cout << "Scanning file: " << scan_path << std::endl;
        auto result = bridge.scanFile(scan_path);
        printDetectionResult(result);
    } else {
        std::cerr << "Invalid scan path: " << scan_path << std::endl;
        return 1;
    }
    
    // Print final statistics
    std::cout << std::endl;
    auto stats = bridge.getStatistics();
    
    std::cout << "╔═ Final Statistics ════════════════════════════════════╗" << std::endl;
    std::cout << "║ Files Scanned:        " << stats.files_scanned << std::endl;
    std::cout << "║ Malware Detected:     " << stats.malware_detected << std::endl;
    std::cout << "║ False Positives:      " << stats.false_positives << std::endl;
    std::cout << "║ False Negatives:      " << stats.false_negatives << std::endl;
    std::cout << "║ Sandbox Analyses:     " << stats.sandbox_analyses << std::endl;
    std::cout << "║ DRL Inferences:       " << stats.drl_inferences << std::endl;
    std::cout << "║ Realtime Detections:  " << stats.realtime_detections << std::endl;
    std::cout << "║ Avg Scan Time:        " << stats.avg_scan_time_ms << " ms" << std::endl;
    std::cout << "║ Avg DRL Time:         " << stats.avg_drl_time_ms << " ms" << std::endl;
    std::cout << "║ Avg Sandbox Time:     " << stats.avg_sandbox_time_ms << " ms" << std::endl;
    
    if (stats.files_scanned > 0) {
        float detection_rate = (float)stats.malware_detected / stats.files_scanned * 100;
        std::cout << "║ Detection Rate:       " << detection_rate << "%" << std::endl;
    }
    
    // Calculate uptime
    auto now = std::chrono::system_clock::now();
    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(now - stats.start_time);
    std::cout << "║ Uptime:               " << uptime.count() << " seconds" << std::endl;
    
    std::cout << "╚═══════════════════════════════════════════════════════╝" << std::endl;
    
    // Shutdown
    std::cout << "\nShutting down..." << std::endl;
    bridge.stop();
    
    std::cout << "\n✅ Malware detection complete. System shutdown." << std::endl;
    
    return 0;
}
