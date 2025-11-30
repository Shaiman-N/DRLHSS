#include "Detection/UnifiedDetectionCoordinator.hpp"
#include <iostream>
#include <fstream>
#include <chrono>

namespace detection {

UnifiedDetectionCoordinator::UnifiedDetectionCoordinator(
    const std::string& model_path,
    const std::string& db_path)
    : model_path_(model_path), db_path_(db_path) {
}

UnifiedDetectionCoordinator::~UnifiedDetectionCoordinator() {
    stop();
}

bool UnifiedDetectionCoordinator::initialize() {
    std::cout << "[UnifiedDetectionCoordinator] Initializing..." << std::endl;
    
    // Initialize database manager
    db_manager_ = std::make_shared<db::DatabaseManager>(db_path_);
    if (!db_manager_->initialize()) {
        std::cerr << "[UnifiedDetectionCoordinator] Failed to initialize database" << std::endl;
        return false;
    }
    
    // Initialize DRL orchestrator
    drl_orchestrator_ = std::make_shared<drl::DRLOrchestrator>(model_path_, db_path_, 16);
    if (!drl_orchestrator_->initialize()) {
        std::cerr << "[UnifiedDetectionCoordinator] Failed to initialize DRL orchestrator" << std::endl;
        return false;
    }
    
    // Initialize NIDPS bridge
    nidps_bridge_ = std::make_shared<NIDPSDetectionBridge>(drl_orchestrator_, db_path_);
    if (!nidps_bridge_->initialize()) {
        std::cerr << "[UnifiedDetectionCoordinator] Failed to initialize NIDPS bridge" << std::endl;
        return false;
    }
    
    // Start pattern learning
    drl_orchestrator_->startPatternLearning();
    
    std::cout << "[UnifiedDetectionCoordinator] Initialized successfully" << std::endl;
    return true;
}

void UnifiedDetectionCoordinator::start() {
    if (running_.load()) {
        std::cout << "[UnifiedDetectionCoordinator] Already running" << std::endl;
        return;
    }
    
    running_.store(true);
    processing_thread_ = std::make_unique<std::thread>(&UnifiedDetectionCoordinator::processingLoop, this);
    
    std::cout << "[UnifiedDetectionCoordinator] Started" << std::endl;
}

void UnifiedDetectionCoordinator::stop() {
    if (!running_.load()) {
        return;
    }
    
    running_.store(false);
    queue_cv_.notify_all();
    
    if (processing_thread_ && processing_thread_->joinable()) {
        processing_thread_->join();
    }
    
    drl_orchestrator_->stopPatternLearning();
    
    std::cout << "[UnifiedDetectionCoordinator] Stopped" << std::endl;
}

drl::DRLOrchestrator::DetectionResponse UnifiedDetectionCoordinator::processNetworkPacket(
    const nidps::PacketPtr& packet) {
    
    total_detections_++;
    network_detections_++;
    
    // Process through NIDPS bridge
    auto response = nidps_bridge_->processPacket(packet);
    
    if (response.is_malicious) {
        malicious_detected_++;
    }
    
    // Create and queue event
    drl::TelemetryData telemetry = nidps_bridge_->convertToTelemetry(packet);
    DetectionEvent event = createEvent(DetectionSource::NETWORK, 
                                      std::to_string(packet->packet_id),
                                      telemetry, response);
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        event_queue_.push(event);
    }
    queue_cv_.notify_one();
    
    return response;
}

drl::DRLOrchestrator::DetectionResponse UnifiedDetectionCoordinator::processFile(
    const std::string& file_path,
    const std::string& file_hash) {
    
    total_detections_++;
    file_detections_++;
    
    // Create telemetry for file
    drl::TelemetryData telemetry;
    telemetry.artifact_hash = file_hash;
    telemetry.sandbox_id = "file_" + file_hash.substr(0, 8);
    telemetry.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    // Process through DRL
    auto response = drl_orchestrator_->processWithDetails(telemetry);
    
    if (response.is_malicious) {
        malicious_detected_++;
    }
    
    // Create and queue event
    DetectionEvent event = createEvent(DetectionSource::FILE, file_path, telemetry, response);
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        event_queue_.push(event);
    }
    queue_cv_.notify_one();
    
    return response;
}

drl::DRLOrchestrator::DetectionResponse UnifiedDetectionCoordinator::processBehavior(
    const drl::TelemetryData& telemetry) {
    
    total_detections_++;
    behavior_detections_++;
    
    // Process through DRL
    auto response = drl_orchestrator_->processWithDetails(telemetry);
    
    if (response.is_malicious) {
        malicious_detected_++;
    }
    
    // Create and queue event
    DetectionEvent event = createEvent(DetectionSource::BEHAVIOR, 
                                      telemetry.sandbox_id,
                                      telemetry, response);
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        event_queue_.push(event);
    }
    queue_cv_.notify_one();
    
    return response;
}

UnifiedDetectionCoordinator::UnifiedStats UnifiedDetectionCoordinator::getStats() const {
    UnifiedStats stats;
    
    stats.total_detections = total_detections_.load();
    stats.network_detections = network_detections_.load();
    stats.file_detections = file_detections_.load();
    stats.behavior_detections = behavior_detections_.load();
    stats.malicious_detected = malicious_detected_.load();
    
    stats.nidps_stats = nidps_bridge_->getStats();
    stats.drl_stats = drl_orchestrator_->getStats();
    stats.db_stats = db_manager_->getStats();
    
    stats.avg_processing_time_ms = stats.drl_stats.avg_inference_time_ms;
    stats.false_positives = stats.drl_stats.false_positives;
    
    return stats;
}

bool UnifiedDetectionCoordinator::exportDetectionEvents(
    const std::string& output_path,
    int limit) {
    
    std::ofstream out(output_path);
    if (!out.is_open()) {
        std::cerr << "[UnifiedDetectionCoordinator] Failed to open output file: " 
                  << output_path << std::endl;
        return false;
    }
    
    // Write header
    out << "timestamp,source,artifact_id,artifact_hash,action,confidence,attack_type,is_malicious\n";
    
    // Query and write events
    auto events = queryRecentEvents(DetectionSource::NETWORK, limit);
    for (const auto& event : events) {
        out << event.timestamp.time_since_epoch().count() << ","
            << static_cast<int>(event.source) << ","
            << event.artifact_id << ","
            << event.artifact_hash << ","
            << event.action << ","
            << event.confidence << ","
            << event.attack_type << ","
            << (event.confidence > 0.7f ? "1" : "0") << "\n";
    }
    
    out.close();
    std::cout << "[UnifiedDetectionCoordinator] Exported " << events.size() 
              << " events to " << output_path << std::endl;
    
    return true;
}

std::vector<UnifiedDetectionCoordinator::DetectionEvent> 
UnifiedDetectionCoordinator::queryRecentEvents(DetectionSource source, int limit) {
    // This is a simplified implementation
    // In production, you'd query from database
    std::vector<DetectionEvent> events;
    
    // For now, return empty vector
    // TODO: Implement database query for detection events
    
    return events;
}

void UnifiedDetectionCoordinator::processingLoop() {
    std::cout << "[UnifiedDetectionCoordinator] Processing loop started" << std::endl;
    
    while (running_.load()) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        queue_cv_.wait(lock, [this] {
            return !event_queue_.empty() || !running_.load();
        });
        
        if (!running_.load()) {
            break;
        }
        
        while (!event_queue_.empty()) {
            DetectionEvent event = event_queue_.front();
            event_queue_.pop();
            lock.unlock();
            
            processEvent(event);
            
            lock.lock();
        }
    }
    
    std::cout << "[UnifiedDetectionCoordinator] Processing loop stopped" << std::endl;
}

void UnifiedDetectionCoordinator::processEvent(const DetectionEvent& event) {
    // Store event in database
    storeEvent(event);
    
    // Log event
    std::cout << "[UnifiedDetectionCoordinator] Event processed - Source: " 
              << static_cast<int>(event.source)
              << " - Action: " << event.action
              << " - Confidence: " << event.confidence << std::endl;
}

void UnifiedDetectionCoordinator::storeEvent(const DetectionEvent& event) {
    // Store telemetry in database
    db_manager_->storeTelemetry(event.telemetry);
    
    // If malicious, store as attack pattern
    if (event.confidence > 0.7f && event.action == 1) {
        drl::AttackPattern pattern;
        pattern.attack_type = event.attack_type;
        pattern.features = event.telemetry.features;
        pattern.confidence = event.confidence;
        pattern.timestamp = event.timestamp;
        pattern.artifact_hash = event.artifact_hash;
        
        db_manager_->storeAttackPattern(pattern);
    }
}

UnifiedDetectionCoordinator::DetectionEvent UnifiedDetectionCoordinator::createEvent(
    DetectionSource source,
    const std::string& artifact_id,
    const drl::TelemetryData& telemetry,
    const drl::DRLOrchestrator::DetectionResponse& response) {
    
    DetectionEvent event;
    event.source = source;
    event.artifact_id = artifact_id;
    event.artifact_hash = telemetry.artifact_hash;
    event.telemetry = telemetry;
    event.action = response.action;
    event.confidence = response.confidence;
    event.attack_type = response.attack_type;
    event.timestamp = std::chrono::system_clock::now();
    
    return event;
}

} // namespace detection

