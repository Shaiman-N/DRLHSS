/**
 * @file AVIntegratedExample.cpp
 * @brief Complete Antivirus integration example with DRLHSS
 * 
 * Demonstrates the fully integrated Antivirus system with:
 * - Real-time file monitoring and scanning
 * - ML-based malware classification
 * - Deep Reinforcement Learning decisions
 * - Cross-platform sandbox analysis
 * - Database persistence
 * - Quarantine management
 */

#include "Detection/AVDetectionBridge.hpp"
#include "Sandbox/SandboxFactory.hpp"
#include <iostream>
#include <csignal>
#include <atomic>
#include <iomanip>
#include <thread>

std::atomic<bool> g_running(true);

void signalHandler(int signal) {
    std::cout << "\\nShutting down Antivirus system gracefully..." << std::endl;
    g_running.store(false);
}

void printAVBanner() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   DRLHSS Antivirus - Integrated Malware Detection System    ║" << std::endl;
    std::cout << "║   Real-time Scanning + ML + DRL + Cross-Platform Sandboxes  ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;
}

void printSystemInfo() {
    std::cout << "System Information:" << std::endl;
    std::cout << "  Platform: " << drlhss::sandbox::SandboxFactory::getPlatformName(
        drlhss::sandbox::SandboxFactory::detectPlatform()) << std::endl;
    
    auto platform = drlhss::sandbox::SandboxFactory::detectPlatform();
    std::cout << "  Sandbox Support: " 
              << (drlhss::sandbox::SandboxFactory::isPlatformSupported(platform) ? "✓ Available" : "✗ Not Available")
              << std::endl;
    
    std::cout << "  Real-time Monitoring: ✓ Enabled" << std::endl;
    std::cout << "  ML Classification: ✓ Enabled" << std::endl;
    std::cout << "  DRL Enhancement: ✓ Enabled" << std::endl;
    std::cout << "  Sandbox Analysis: ✓ Enabled" << std::endl;
    std::cout << std::endl;
}

void printScanResult(const drlhss::detection::AVDetectionBridge::IntegratedScanResult& result) {
    std::cout << "\\n┌─ File Scan Result ─────────────────────────────────────┐" << std::endl;
    std::cout << "│ File: " << result.file_path << std::endl;
    std::cout << "│ Hash: " << result.file_hash.substr(0, 16) << "..." << std::endl;
    std::cout << "│ Size: " << result.file_size << " bytes" << std::endl;
    std::cout << "│ Type: " << result.file_type << std::endl;
    std::cout << "│" << std::endl;
    
    // Detection results
    std::cout << "│ AV Confidence: " << std::fixed << std::setprecision(1) 
              << (result.av_confidence * 100) << "%" << std::endl;
    
    std::cout << "│ ML Confidence: " << (result.ml_confidence * 100) << "%" << std::endl;
    
    if (result.drl_action >= 0) {
        std::cout << "│ DRL Action: " << result.drl_action 
                  << " (0=Allow, 1=Block, 2=Quarantine, 3=DeepScan)" << std::endl;
        std::cout << "│ DRL Confidence: " << (result.drl_confidence * 100) << "%" << std::endl;
    }
    
    std::cout << "│" << std::endl;
    
    // Final decision
    if (result.is_malicious) {
        std::cout << "│ ⚠️  THREAT DETECTED: " << result.threat_classification << std::endl;
        std::cout << "│ 🛡️  Action: " << result.recommended_action << std::endl;
    } else {
        std::cout << "│ ✅ File is CLEAN" << std::endl;
    }
    
    // Sandbox results
    if (result.sandbox_result.success) {
        std::cout << "│" << std::endl;
        std::cout << "│ Sandbox Analysis:" << std::endl;
        std::cout << "│   Threat Score: " << result.sandbox_result.threat_score << "/100" << std::endl;
        std::cout << "│   File Modified: " << (result.sandbox_result.file_system_modified ? "Yes" : "No") << std::endl;
        std::cout << "│   Network Activity: " << (result.sandbox_result.network_activity_detected ? "Yes" : "No") << std::endl;
        std::cout << "│   Process Created: " << (result.sandbox_result.process_created ? "Yes" : "No") << std::endl;
        std::cout << "│   Execution Time: " << result.sandbox_result.execution_time.count() << "ms" << std::endl;
    }
    
    // Indicators
    if (!result.indicators.empty()) {
        std::cout << "│" << std::endl;
        std::cout << "│ Indicators:" << std::endl;
        for (const auto& indicator : result.indicators) {
            std::cout << "│   - " << indicator << std::endl;
        }
    }
    
    std::cout << "│ Scan Duration: " << result.scan_duration.count() << "ms" << std::endl;
    std::cout << "└────────────────────────────────────────────────────────┘" << std::endl;
}

void printStatistics(const drlhss::detection::AVDetectionBridge::AVStatistics& stats) {
    std::cout << "\\n╔═ Antivirus Statistics ════════════════════════════════╗" << std::endl;
    std::cout << "║ Files Scanned:        " << stats.files_scanned << std::endl;
    std::cout << "║ Threats Detected:     " << stats.threats_detected << std::endl;
    std::cout << "║ Files Quarantined:    " << stats.files_quarantined << std::endl;
    std::cout << "║ False Positives:      " << stats.false_positives << std::endl;
    std::cout << "║ Sandbox Analyses:     " << stats.sandbox_analyses << std::endl;
    std::cout << "║ DRL Inferences:       " << stats.drl_inferences << std::endl;
    std::cout << "║ ML Detections:        " << stats.ml_detections << std::endl;
    std::cout << "║ Avg Scan Time:       " << std::fixed << std::setprecision(2) 
              << stats.avg_scan_time_ms << " ms" << std::endl;
    
    if (stats.files_scanned > 0) {
        float detection_rate = (float)stats.threats_detected / stats.files_scanned * 100;
        std::cout << "║ Detection Rate:       " << std::fixed << std::setprecision(1) 
                  << detection_rate << "%" << std::endl;
    }
    
    std::cout << "╚═══════════════════════════════════════════════════════╝" << std::endl;
}

int main(int argc, char** argv) {
    // Setup signal handler
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    
    printAVBanner();
    printSystemInfo();
    
    // Parse command line arguments
    std::string scan_path = ".";
    bool real_time_mode = false;  // Default to scan mode
    
    if (argc > 1) {
        scan_path = argv[1];
    }
    
    if (argc > 2 && std::string(argv[2]) == "--realtime") {
        real_time_mode = true;
    }
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Scan Path: " << scan_path << std::endl;
    std::cout << "  Real-time Mode: " << (real_time_mode ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  ML Model: models/onnx/antivirus_static_model.onnx" << std::endl;
    std::cout << "  DRL Model: models/onnx/dqn_model.onnx" << std::endl;
    std::cout << "  Database: data/drlhss.db" << std::endl;
    std::cout << "  Quarantine: quarantine/" << std::endl;
    std::cout << std::endl;
    
    // Configure AV bridge
    drlhss::detection::AVDetectionBridge::AVBridgeConfig config;
    config.ml_model_path = "models/onnx/antivirus_static_model.onnx";
    config.drl_model_path = "models/onnx/dqn_model.onnx";
    config.database_path = "data/drlhss.db";
    config.quarantine_path = "quarantine/";
    config.malware_threshold = 0.7f;
    config.enable_real_time_monitoring = real_time_mode;
    config.enable_sandbox_analysis = true;
    config.enable_drl_inference = true;
    config.enable_dynamic_analysis = false;  // Disabled by default for performance
    
    // Set scan directories based on platform
    auto platform = drlhss::sandbox::SandboxFactory::detectPlatform();
    if (platform == drlhss::sandbox::Platform::WINDOWS) {
        config.scan_directories = {"C:\\\\Users", "C:\\\\Temp", "C:\\\\Windows\\\\Temp"};
    } else if (platform == drlhss::sandbox::Platform::MACOS) {
        config.scan_directories = {"/Users", "/tmp", "/Applications"};
    } else {
        config.scan_directories = {"/home", "/tmp", "/var/tmp"};
    }
    
    // Create and initialize AV bridge
    std::cout << "[1/4] Initializing Antivirus Detection Bridge..." << std::endl;
    drlhss::detection::AVDetectionBridge av_bridge(config);
    
    if (!av_bridge.initialize()) {
        std::cerr << "✗ Failed to initialize AV bridge!" << std::endl;
        return 1;
    }
    std::cout << "✓ AV bridge initialized successfully" << std::endl;
    
    // Set scan callback
    std::cout << "[2/4] Setting up scan callback..." << std::endl;
    av_bridge.setScanCallback(printScanResult);
    std::cout << "✓ Scan callback configured" << std::endl;
    
    // Update signatures
    std::cout << "[3/4] Updating signature database..." << std::endl;
    if (av_bridge.updateSignatures()) {
        std::cout << "✓ Signatures updated" << std::endl;
    } else {
        std::cout << "⚠ Signature update failed (continuing anyway)" << std::endl;
    }
    
    // Start monitoring or perform scan
    if (real_time_mode) {
        std::cout << "[4/4] Starting real-time monitoring..." << std::endl;
        av_bridge.startMonitoring();
        std::cout << "✓ Real-time monitoring started" << std::endl;
        std::cout << std::endl;
        
        std::cout << "═══════════════════════════════════════════════════════════" << std::endl;
        std::cout << "  Antivirus is now monitoring file system changes..." << std::endl;
        std::cout << "  Press Ctrl+C to stop" << std::endl;
        std::cout << "═══════════════════════════════════════════════════════════" << std::endl;
        std::cout << std::endl;
        
        // Main loop - print statistics periodically
        int stats_interval = 0;
        while (g_running.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            stats_interval++;
            
            // Print statistics every 60 seconds
            if (stats_interval >= 60) {
                stats_interval = 0;
                printStatistics(av_bridge.getStatistics());
                std::cout << std::endl;
            }
        }
        
        // Stop monitoring
        av_bridge.stopMonitoring();
        
    } else {
        std::cout << "[4/4] Performing directory scan..." << std::endl;
        std::cout << "✓ Starting scan of: " << scan_path << std::endl;
        std::cout << std::endl;
        
        std::cout << "═══════════════════════════════════════════════════════════" << std::endl;
        std::cout << "  Scanning directory: " << scan_path << std::endl;
        std::cout << "═══════════════════════════════════════════════════════════" << std::endl;
        
        // Perform directory scan
        auto scan_results = av_bridge.scanDirectory(scan_path);
        
        std::cout << "\\nScan completed. Results:" << std::endl;
        
        int clean_files = 0;
        int malicious_files = 0;
        
        for (const auto& result : scan_results) {
            if (result.is_malicious) {
                malicious_files++;
                printScanResult(result);
            } else {
                clean_files++;
            }
        }
        
        std::cout << "\\n╔═ Scan Summary ════════════════════════════════════════╗" << std::endl;
        std::cout << "║ Total Files Scanned: " << scan_results.size() << std::endl;
        std::cout << "║ Clean Files:         " << clean_files << std::endl;
        std::cout << "║ Malicious Files:     " << malicious_files << std::endl;
        std::cout << "╚═══════════════════════════════════════════════════════╝" << std::endl;
    }
    
    // Print final statistics
    std::cout << "\\nFinal Statistics:" << std::endl;
    printStatistics(av_bridge.getStatistics());
    
    std::cout << "\\nAntivirus system shutdown complete. Stay safe!" << std::endl;
    
    return 0;
}
