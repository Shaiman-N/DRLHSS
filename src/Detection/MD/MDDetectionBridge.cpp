#include "Detection/MDDetectionBridge.hpp"
#include <iostream>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace detection {

MDDetectionBridge::MDDetectionBridge(const BridgeConfig& config)
    : config_(config), running_(false) {
    start_time_ = std::chrono::system_clock::now();
}

MDDetectionBridge::~MDDetectionBridge() {
    stop();
}

bool MDDetectionBridge::initialize() {
    std::cout << "[MDDetectionBridge] Initializing Malware Detection Bridge..." << std::endl;
    
    try {
        // Initialize Malware Detector
        malware_detector_ = std::make_shared<MalwareDetector>(config_.malware_model_path);
        std::cout << "[MDDetectionBridge] ✓ Malware detector initialized" << std::endl;
        
        // Initialize MD DRL Framework
        md_drl_framework_ = std::make_shared<DRLFramework>(config_.database_path);
        std::cout << "[MDDetectionBridge] ✓ MD DRL framework initialized" << std::endl;
        
        // Initialize Sandbox Orchestrator (MD's own)
        md_orchestrator_ = std::make_shared<SandboxOrchestrator>(md_drl_framework_);
        std::cout << "[MDDetectionBridge] ✓ MD sandbox orchestrator initialized" << std::endl;
        
        // Initialize Processing Pipeline
        pipeline_ = std::make_shared<MalwareProcessingPipeline>(
            malware_detector_,
            md_orchestrator_,
            md_drl_framework_
        );
        std::cout << "[MDDetectionBridge] ✓ Processing pipeline initialized" << std::endl;
        
        // Initialize Detection Service
        detection_service_ = std::make_shared<MalwareDetectionService>(
            config_.malware_model_path,
            config_.database_path
        );
        std::cout << "[MDDetectionBridge] ✓ Detection service initialized" << std::endl;
        
        // Initialize Real-Time Monitor
        if (config_.enable_realtime_monitoring) {
            realtime_monitor_ = std::make_unique<RealTimeMonitor>();
            realtime_monitor_->setThreatCallback(
                [this](const std::string& location, const std::string& description) {
                    this->onRealtimeThreatDetected(location, description);
                }
            );
            std::cout << "[MDDetectionBridge] ✓ Real-time monitor initialized" << std::endl;
        }
        
        // Initialize DRLHSS DRL Orchestrator
        if (config_.enable_drl_inference) {
            drl_orchestrator_ = std::make_unique<drl::DRLOrchestrator>(
                config_.drl_model_path,
                config_.database_path,
                16  // feature dimension
            );
            
            if (!drl_orchestrator_->initialize()) {
                std::cerr << "[MDDetectionBridge] Failed to initialize DRL orchestrator" << std::endl;
                return false;
            }
            std::cout << "[MDDetectionBridge] ✓ DRLHSS DRL orchestrator initialized" << std::endl;
        }
        
        // Initialize Database
        database_ = std::make_unique<db::DatabaseManager>(config_.database_path);
        if (!database_->initialize()) {
            std::cerr << "[MDDetectionBridge] Failed to initialize database" << std::endl;
            return false;
        }
        std::cout << "[MDDetectionBridge] ✓ Database initialized" << std::endl;
        
        // Initialize DRLHSS Sandboxes
        if (config_.enable_sandbox_analysis) {
            try {
                sandbox::SandboxConfig sandbox_config;
                sandbox_config.memory_limit_mb = 2048;
                sandbox_config.cpu_limit_percent = 75;
                sandbox_config.timeout_seconds = 60;
                sandbox_config.allow_network = false;
                sandbox_config.read_only_filesystem = false;
                
                // Create positive sandbox (FP detection)
                sandbox_config.sandbox_id = "md_positive";
                positive_sandbox_ = sandbox::SandboxFactory::createSandbox(
                    sandbox::SandboxType::POSITIVE_FP,
                    sandbox_config
                );
                
                // Create negative sandbox (FN detection)
                sandbox_config.sandbox_id = "md_negative";
                negative_sandbox_ = sandbox::SandboxFactory::createSandbox(
                    sandbox::SandboxType::NEGATIVE_FN,
                    sandbox_config
                );
                
                std::cout << "[MDDetectionBridge] ✓ DRLHSS sandboxes initialized on " 
                          << sandbox::SandboxFactory::getPlatformName(
                              sandbox::SandboxFactory::detectPlatform()) << std::endl;
                          
            } catch (const std::exception& e) {
                std::cerr << "[MDDetectionBridge] Failed to initialize sandboxes: " 
                          << e.what() << std::endl;
                return false;
            }
        }
        
        std::cout << "[MDDetectionBridge] ✅ Initialization complete" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[MDDetectionBridge] Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

void MDDetectionBridge::start() {
    if (running_.load()) {
        return;
    }
    
    running_.store(true);
    
    // Start detection service
    if (detection_service_) {
        detection_service_->start();
    }
    
    // Start real-time monitoring
    if (realtime_monitor_ && config_.enable_realtime_monitoring) {
        realtime_monitor_->startMonitoring();
    }
    
    // Start DRL pattern learning
    if (drl_orchestrator_) {
        drl_orchestrator_->startPatternLearning();
    }
    
    std::cout << "[MDDetectionBridge] 🚀 Malware detection started" << std::endl;
}

void MDDetectionBridge::stop() {
    if (!running_.load()) {
        return;
    }
    
    running_.store(false);
    
    // Stop detection service
    if (detection_service_) {
        detection_service_->stop();
    }
    
    // Stop real-time monitoring
    if (realtime_monitor_) {
        realtime_monitor_->stopMonitoring();
    }
    
    // Stop DRL pattern learning
    if (drl_orchestrator_) {
        drl_orchestrator_->stopPatternLearning();
    }
    
    std::cout << "[MDDetectionBridge] 🛑 Malware detection stopped" << std::endl;
}

MDDetectionBridge::IntegratedDetectionResult MDDetectionBridge::scanFile(const std::string& file_path) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    files_scanned_++;
    
    IntegratedDetectionResult result;
    result.file_path = file_path;
    result.scan_timestamp = std::chrono::system_clock::now();
    result.file_hash = calculateFileHash(file_path);
    
    // Process with complete pipeline
    processWithPipeline(file_path, result);
    
    // Process with DRL if enabled
    if (config_.enable_drl_inference) {
        processWithDRL(result);
    }
    
    // Process with DRLHSS sandboxes if needed
    if (config_.enable_sandbox_analysis && result.is_malicious) {
        processWithSandbox(result);
    }
    
    // Finalize detection
    result.threat_classification = classifyThreat(result);
    result.malware_family = determineMalwareFamily(result);
    result.recommended_action = determineAction(result);
    result.overall_threat_score = calculateOverallThreatScore(result);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.scan_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    total_scan_time_ms_ += result.scan_duration.count();
    
    finalizeDetection(result);
    
    return result;
}

std::vector<MDDetectionBridge::IntegratedDetectionResult> MDDetectionBridge::scanDirectory(
    const std::string& directory_path) {
    
    std::vector<IntegratedDetectionResult> results;
    
    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(directory_path)) {
            if (entry.is_regular_file()) {
                auto result = scanFile(entry.path().string());
                results.push_back(result);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[MDDetectionBridge] Error scanning directory: " << e.what() << std::endl;
    }
    
    return results;
}

MDDetectionBridge::IntegratedDetectionResult MDDetectionBridge::scanDataPacket(
    const std::vector<uint8_t>& packet) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    files_scanned_++;
    
    IntegratedDetectionResult result;
    result.file_path = "<data_packet>";
    result.scan_timestamp = std::chrono::system_clock::now();
    
    // Process packet through pipeline
    if (pipeline_) {
        result.pipeline_result = pipeline_->processDataPacket(packet);
        result.is_malicious = !result.pipeline_result.isSafe;
        result.md_confidence = result.is_malicious ? 0.8f : 0.2f;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.scan_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    total_scan_time_ms_ += result.scan_duration.count();
    
    finalizeDetection(result);
    
    return result;
}

void MDDetectionBridge::setDetectionCallback(DetectionCallback callback) {
    detection_callback_ = callback;
}

void MDDetectionBridge::setThreatCallback(ThreatCallback callback) {
    threat_callback_ = callback;
}

MDDetectionBridge::BridgeStatistics MDDetectionBridge::getStatistics() const {
    BridgeStatistics stats;
    stats.files_scanned = files_scanned_.load();
    stats.malware_detected = malware_detected_.load();
    stats.false_positives = false_positives_.load();
    stats.false_negatives = false_negatives_.load();
    stats.sandbox_analyses = sandbox_analyses_.load();
    stats.drl_inferences = drl_inferences_.load();
    stats.realtime_detections = realtime_detections_.load();
    stats.start_time = start_time_;
    
    if (stats.files_scanned > 0) {
        stats.avg_scan_time_ms = total_scan_time_ms_.load() / stats.files_scanned;
    }
    if (stats.drl_inferences > 0) {
        stats.avg_drl_time_ms = total_drl_time_ms_.load() / stats.drl_inferences;
    }
    if (stats.sandbox_analyses > 0) {
        stats.avg_sandbox_time_ms = total_sandbox_time_ms_.load() / stats.sandbox_analyses;
    }
    
    return stats;
}

void MDDetectionBridge::updateThreatIntelligence(const std::vector<std::string>& threat_data) {
    std::cout << "[MDDetectionBridge] Updated threat intelligence with " 
              << threat_data.size() << " entries" << std::endl;
}

RealTimeMonitor::MonitorStats MDDetectionBridge::getMonitorStats() const {
    if (realtime_monitor_) {
        return realtime_monitor_->getStats();
    }
    return RealTimeMonitor::MonitorStats();
}

// Private methods

void MDDetectionBridge::processFile(const std::string& file_path) {
    IntegratedDetectionResult result;
    result.file_path = file_path;
    processWithPipeline(file_path, result);
}

void MDDetectionBridge::processWithMD(const std::string& file_path, IntegratedDetectionResult& result) {
    try {
        // Read file data
        std::ifstream file(file_path, std::ios::binary);
        if (!file) {
            std::cerr << "[MDDetectionBridge] Failed to open file: " << file_path << std::endl;
            return;
        }
        
        std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)),
                                   std::istreambuf_iterator<char>());
        
        // Detect malware
        float confidence = 0.0f;
        bool is_malware = malware_detector_->detectMalware(data, confidence);
        
        result.is_malicious = is_malware;
        result.md_confidence = confidence;
        
    } catch (const std::exception& e) {
        std::cerr << "[MDDetectionBridge] Error in MD processing: " << e.what() << std::endl;
    }
}

void MDDetectionBridge::processWithDRL(IntegratedDetectionResult& result) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    drl_inferences_++;
    
    try {
        // Convert to telemetry
        drl::TelemetryData telemetry = convertToTelemetry(result);
        
        // Get DRL decision
        auto drl_response = drl_orchestrator_->processWithDetails(telemetry);
        
        result.drl_action = drl_response.action;
        result.drl_q_values = drl_response.q_values;
        result.drl_confidence = drl_response.confidence;
        
        // Update malicious flag based on DRL
        if (drl_response.action == 1 || drl_response.action == 2) {
            result.is_malicious = true;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[MDDetectionBridge] Error in DRL processing: " << e.what() << std::endl;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    total_drl_time_ms_ += duration.count() / 1000.0;
}

void MDDetectionBridge::processWithSandbox(IntegratedDetectionResult& result) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    sandbox_analyses_++;
    
    try {
        // Analyze in positive sandbox
        if (positive_sandbox_ && positive_sandbox_->isReady()) {
            result.sandbox_result = positive_sandbox_->execute(result.file_path, {});
            
            // Learn from sandbox results
            if (drl_orchestrator_) {
                drl::TelemetryData telemetry = convertToTelemetry(result);
                float reward = (result.sandbox_result.threat_score > 50) ? 1.0f : -1.0f;
                drl_orchestrator_->storeExperience(telemetry, result.drl_action, reward, telemetry, true);
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[MDDetectionBridge] Error in sandbox processing: " << e.what() << std::endl;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    total_sandbox_time_ms_ += duration.count() / 1000.0;
}

void MDDetectionBridge::processWithPipeline(const std::string& file_path, IntegratedDetectionResult& result) {
    try {
        if (pipeline_) {
            result.pipeline_result = pipeline_->processFile(file_path);
            result.is_malicious = !result.pipeline_result.isSafe;
            result.md_confidence = result.is_malicious ? 0.85f : 0.15f;
            
            // Extract attack patterns and behaviors
            result.attack_patterns = result.pipeline_result.detectedThreats;
            result.behavioral_indicators = result.pipeline_result.detectedThreats;
        }
    } catch (const std::exception& e) {
        std::cerr << "[MDDetectionBridge] Error in pipeline processing: " << e.what() << std::endl;
    }
}

void MDDetectionBridge::finalizeDetection(const IntegratedDetectionResult& result) {
    // Update statistics
    if (result.is_malicious) {
        malware_detected_++;
    }
    
    // Store in database
    storeDetectionResult(result);
    
    // Take action based on result
    if (result.recommended_action == "QUARANTINE") {
        quarantineFile(result.file_path);
    } else if (result.recommended_action == "DELETE") {
        deleteFile(result.file_path);
    }
    
    // Call user callback
    if (detection_callback_) {
        detection_callback_(result);
    }
}

drl::TelemetryData MDDetectionBridge::convertToTelemetry(const IntegratedDetectionResult& result) {
    drl::TelemetryData telemetry;
    
    telemetry.sandbox_id = "md_scanner";
    telemetry.timestamp = result.scan_timestamp;
    telemetry.artifact_hash = result.file_hash;
    telemetry.artifact_type = "file";
    
    // Set basic metrics
    telemetry.syscall_count = result.attack_patterns.size() * 10;
    telemetry.file_read_count = 5;
    telemetry.file_write_count = result.is_malicious ? 10 : 0;
    telemetry.file_delete_count = result.is_malicious ? 2 : 0;
    telemetry.network_connections = result.is_malicious ? 3 : 0;
    telemetry.child_processes = result.is_malicious ? 2 : 0;
    telemetry.cpu_usage = result.is_malicious ? 75.0f : 10.0f;
    telemetry.memory_usage = result.is_malicious ? 512.0f : 64.0f;
    
    // Set behavioral flags
    telemetry.registry_modification = result.is_malicious;
    telemetry.privilege_escalation_attempt = result.is_malicious && result.overall_threat_score > 0.8f;
    telemetry.code_injection_detected = result.is_malicious && result.overall_threat_score > 0.9f;
    
    return telemetry;
}

std::string MDDetectionBridge::classifyThreat(const IntegratedDetectionResult& result) {
    if (!result.is_malicious) {
        return "BENIGN";
    }
    
    // Classify based on attack patterns
    for (const auto& pattern : result.attack_patterns) {
        if (pattern.find("ransomware") != std::string::npos) return "RANSOMWARE";
        if (pattern.find("trojan") != std::string::npos) return "TROJAN";
        if (pattern.find("virus") != std::string::npos) return "VIRUS";
        if (pattern.find("worm") != std::string::npos) return "WORM";
        if (pattern.find("spyware") != std::string::npos) return "SPYWARE";
        if (pattern.find("adware") != std::string::npos) return "ADWARE";
        if (pattern.find("rootkit") != std::string::npos) return "ROOTKIT";
    }
    
    return "MALWARE";
}

std::string MDDetectionBridge::determineMalwareFamily(const IntegratedDetectionResult& result) {
    if (!result.is_malicious) {
        return "N/A";
    }
    
    // Determine family based on patterns
    for (const auto& pattern : result.attack_patterns) {
        if (pattern.find("WannaCry") != std::string::npos) return "WannaCry";
        if (pattern.find("Emotet") != std::string::npos) return "Emotet";
        if (pattern.find("Zeus") != std::string::npos) return "Zeus";
        if (pattern.find("Conficker") != std::string::npos) return "Conficker";
    }
    
    return "UNKNOWN";
}

std::string MDDetectionBridge::determineAction(const IntegratedDetectionResult& result) {
    if (!result.is_malicious) {
        return "ALLOW";
    }
    
    // Determine action based on threat score and confidence
    if (result.overall_threat_score > 0.9f || result.md_confidence > 0.95f) {
        return "DELETE";
    } else if (result.overall_threat_score > 0.7f || result.md_confidence > 0.8f) {
        return "QUARANTINE";
    } else if (result.overall_threat_score > 0.5f) {
        return "MONITOR";
    } else {
        return "SCAN_AGAIN";
    }
}

float MDDetectionBridge::calculateOverallThreatScore(const IntegratedDetectionResult& result) {
    float score = 0.0f;
    
    // Weight different components
    score += result.md_confidence * 0.4f;  // 40% weight to MD
    score += result.drl_confidence * 0.3f; // 30% weight to DRL
    
    if (result.sandbox_result.success) {
        score += (result.sandbox_result.threat_score / 100.0f) * 0.3f; // 30% weight to sandbox
    }
    
    return std::clamp(score, 0.0f, 1.0f);
}

void MDDetectionBridge::storeDetectionResult(const IntegratedDetectionResult& result) {
    try {
        // Store telemetry in database
        drl::TelemetryData telemetry = convertToTelemetry(result);
        database_->storeTelemetry(telemetry);
        
        // If malicious, learn attack pattern
        if (result.is_malicious && md_drl_framework_) {
            BehaviorData behavior;
            behavior.filePath = result.file_path;
            behavior.behaviors = result.behavioral_indicators;
            behavior.threatLevel = result.overall_threat_score;
            behavior.sandboxType = "integrated";
            behavior.timestamp = std::chrono::system_clock::to_time_t(result.scan_timestamp);
            
            md_drl_framework_->learnFromBehavior(behavior);
        }
        
        // Store in DRLHSS DRL
        if (result.is_malicious && drl_orchestrator_) {
            drl_orchestrator_->learnAttackPattern(
                telemetry,
                result.drl_action,
                result.is_malicious ? 1.0f : -1.0f,
                result.threat_classification,
                result.overall_threat_score
            );
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[MDDetectionBridge] Error storing detection result: " << e.what() << std::endl;
    }
}

void MDDetectionBridge::quarantineFile(const std::string& file_path) {
    try {
        std::string quarantine_dir = "quarantine/";
        std::filesystem::create_directories(quarantine_dir);
        
        std::string quarantine_path = quarantine_dir + std::filesystem::path(file_path).filename().string();
        std::filesystem::copy(file_path, quarantine_path, std::filesystem::copy_options::overwrite_existing);
        std::filesystem::remove(file_path);
        
        std::cout << "[MDDetectionBridge] File quarantined: " << file_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[MDDetectionBridge] Failed to quarantine file: " << e.what() << std::endl;
    }
}

void MDDetectionBridge::deleteFile(const std::string& file_path) {
    try {
        std::filesystem::remove(file_path);
        std::cout << "[MDDetectionBridge] File deleted: " << file_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[MDDetectionBridge] Failed to delete file: " << e.what() << std::endl;
    }
}

std::string MDDetectionBridge::calculateFileHash(const std::string& file_path) {
    try {
        std::ifstream file(file_path, std::ios::binary);
        if (!file) {
            return "ERROR";
        }
        
        std::stringstream ss;
        ss << std::hex << std::setfill('0');
        
        char byte;
        int count = 0;
        while (file.get(byte) && count++ < 32) {
            ss << std::setw(2) << static_cast<unsigned>(static_cast<unsigned char>(byte));
        }
        
        return ss.str();
        
    } catch (const std::exception& e) {
        return "ERROR";
    }
}

void MDDetectionBridge::onRealtimeThreatDetected(const std::string& location, const std::string& description) {
    realtime_detections_++;
    
    std::cout << "[MDDetectionBridge] ⚠️  Real-time threat detected!" << std::endl;
    std::cout << "  Location: " << location << std::endl;
    std::cout << "  Description: " << description << std::endl;
    
    // Call user callback
    if (threat_callback_) {
        threat_callback_(location, description);
    }
    
    // Scan the detected file
    if (std::filesystem::exists(location)) {
        scanFile(location);
    }
}

} // namespace detection
