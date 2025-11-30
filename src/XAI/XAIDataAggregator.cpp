/**
 * @file XAIDataAggregator.cpp
 * @brief Implementation of XAI Data Aggregator
 */

#include "XAI/XAIDataAggregator.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>

namespace xai {

XAIDataAggregator::XAIDataAggregator(const std::string& db_path)
    : db_path_(db_path) {
    
    // Initialize component status
    component_status_["AV"] = "UNKNOWN";
    component_status_["NIDPS"] = "UNKNOWN";
    component_status_["DRL"] = "UNKNOWN";
    component_status_["Sandbox"] = "UNKNOWN";
    component_status_["Telemetry"] = "UNKNOWN";
    component_status_["Database"] = "UNKNOWN";
}

XAIDataAggregator::~XAIDataAggregator() {
    stopStreaming();
}

bool XAIDataAggregator::initialize() {
    try {
        // Initialize database manager
        db_manager_ = std::make_unique<db::DatabaseManager>(db_path_);
        
        if (!db_manager_->initialize()) {
            std::cerr << "[XAI Aggregator] Failed to initialize database" << std::endl;
            return false;
        }
        
        component_status_["Database"] = "RUNNING";
        
        std::cout << "[XAI Aggregator] Initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[XAI Aggregator] Initialization error: " << e.what() << std::endl;
        return false;
    }
}

void XAIDataAggregator::startStreaming() {
    if (streaming_active_.load()) {
        return;
    }
    
    streaming_active_.store(true);
    streaming_thread_ = std::make_unique<std::thread>(&XAIDataAggregator::streamingLoop, this);
    
    std::cout << "[XAI Aggregator] Started real-time streaming" << std::endl;
}

void XAIDataAggregator::stopStreaming() {
    if (!streaming_active_.load()) {
        return;
    }
    
    streaming_active_.store(false);
    
    if (streaming_thread_ && streaming_thread_->joinable()) {
        streaming_thread_->join();
    }
    
    std::cout << "[XAI Aggregator] Stopped real-time streaming" << std::endl;
}

// ========== Real-Time Event Streaming ==========

void XAIDataAggregator::registerEventCallback(EventStreamCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    event_callbacks_.push_back(callback);
}

void XAIDataAggregator::registerThreatCallback(ThreatDetectionCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    threat_callbacks_.push_back(callback);
}

void XAIDataAggregator::ingestEvent(const telemetry::TelemetryEvent& event) {
    // Add to queue
    {
        std::lock_guard<std::mutex> lock(event_queue_mutex_);
        event_queue_.push(event);
    }
    
    // Update cache
    updateRecentEventsCache(event);
    
    events_ingested_.fetch_add(1);
}

void XAIDataAggregator::ingestThreat(const ThreatInfo& threat) {
    // Add to queue
    {
        std::lock_guard<std::mutex> lock(threat_queue_mutex_);
        threat_queue_.push(threat);
    }
    
    // Update cache
    updateRecentThreatsCache(threat);
    
    threats_ingested_.fetch_add(1);
}

// ========== Batch Data Retrieval ==========

std::vector<telemetry::TelemetryEvent> XAIDataAggregator::getRecentEvents(
    int limit,
    telemetry::EventType event_type
) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    std::vector<telemetry::TelemetryEvent> result;
    
    for (const auto& event : recent_events_cache_) {
        if (event_type == telemetry::EventType::UNKNOWN || event.type == event_type) {
            result.push_back(event);
            if (result.size() >= static_cast<size_t>(limit)) {
                break;
            }
        }
    }
    
    return result;
}

std::vector<telemetry::TelemetryEvent> XAIDataAggregator::getEventsByTimeRange(
    std::chrono::system_clock::time_point start_time,
    std::chrono::system_clock::time_point end_time
) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    std::vector<telemetry::TelemetryEvent> result;
    
    for (const auto& event : recent_events_cache_) {
        if (event.timestamp >= start_time && event.timestamp <= end_time) {
            result.push_back(event);
        }
    }
    
    return result;
}

std::vector<telemetry::TelemetryEvent> XAIDataAggregator::getEventsByProcess(
    int pid,
    int limit
) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    std::vector<telemetry::TelemetryEvent> result;
    
    for (const auto& event : recent_events_cache_) {
        if (event.pid == pid) {
            result.push_back(event);
            if (result.size() >= static_cast<size_t>(limit)) {
                break;
            }
        }
    }
    
    return result;
}

std::vector<ThreatInfo> XAIDataAggregator::getRecentThreats(int limit) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    std::vector<ThreatInfo> result;
    
    for (const auto& threat : recent_threats_cache_) {
        result.push_back(threat);
        if (result.size() >= static_cast<size_t>(limit)) {
            break;
        }
    }
    
    return result;
}

std::vector<ThreatInfo> XAIDataAggregator::getThreatsBySeverity(
    ThreatSeverity severity,
    int limit
) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    std::vector<ThreatInfo> result;
    
    for (const auto& threat : recent_threats_cache_) {
        if (threat.severity >= severity) {
            result.push_back(threat);
            if (result.size() >= static_cast<size_t>(limit)) {
                break;
            }
        }
    }
    
    return result;
}

std::vector<drl::AttackPattern> XAIDataAggregator::getAttackPatterns(
    const std::string& attack_type,
    int limit
) {
    if (!db_manager_) {
        return {};
    }
    
    if (attack_type.empty()) {
        // Get all patterns
        return db_manager_->queryAttackPatterns("", limit);
    } else {
        return db_manager_->queryAttackPatterns(attack_type, limit);
    }
}

// ========== System State Queries ==========

SystemSnapshot XAIDataAggregator::getSystemSnapshot() {
    SystemSnapshot snapshot;
    
    snapshot.snapshot_time = std::chrono::system_clock::now();
    
    // Get metrics
    auto metrics = getTodayMetrics();
    snapshot.threats_today = metrics.total_threats;
    snapshot.active_alerts = metrics.blocked_threats;
    
    // Get health status
    snapshot.health_status = getHealthStatus();
    
    // Get DRL confidence
    snapshot.drl_confidence = getDRLConfidence();
    
    // Get component status
    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        snapshot.component_status = component_status_;
    }
    
    // Get recent events and threats
    snapshot.recent_events = getRecentEvents(100);
    snapshot.recent_threats = getRecentThreats(50);
    
    // Get performance metrics
    auto stats = getStats();
    snapshot.metrics.events_processed = stats.events_ingested;
    snapshot.metrics.threats_detected = stats.threats_ingested;
    
    return snapshot;
}

std::string XAIDataAggregator::getComponentStatus(const std::string& component_name) {
    std::lock_guard<std::mutex> lock(status_mutex_);
    
    auto it = component_status_.find(component_name);
    if (it != component_status_.end()) {
        return it->second;
    }
    
    return "UNKNOWN";
}

void XAIDataAggregator::updateComponentStatus(
    const std::string& component_name,
    const std::string& status
) {
    std::lock_guard<std::mutex> lock(status_mutex_);
    component_status_[component_name] = status;
}

XAIDataAggregator::ThreatMetrics XAIDataAggregator::getTodayMetrics() {
    return computeTodayMetrics();
}

float XAIDataAggregator::getDRLConfidence() {
    std::lock_guard<std::mutex> lock(confidence_mutex_);
    return drl_confidence_;
}

void XAIDataAggregator::updateDRLConfidence(float confidence) {
    std::lock_guard<std::mutex> lock(confidence_mutex_);
    drl_confidence_ = confidence;
}

// ========== Threat Metrics Aggregation ==========

XAIDataAggregator::AggregatedStats XAIDataAggregator::aggregateStats(int time_window_hours) {
    AggregatedStats stats;
    
    auto now = std::chrono::system_clock::now();
    auto start_time = now - std::chrono::hours(time_window_hours);
    
    // Get events in time window
    auto events = getEventsByTimeRange(start_time, now);
    stats.total_events = events.size();
    
    // Get threats in time window
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    for (const auto& threat : recent_threats_cache_) {
        // Count threats
        stats.total_threats++;
        
        // Count by type
        stats.top_threat_types[threat.threat_type]++;
        
        // Count by process
        stats.top_processes[threat.process_name]++;
    }
    
    // Compute rates
    if (stats.total_events > 0) {
        stats.detection_rate = static_cast<float>(stats.total_threats) / stats.total_events;
    }
    
    return stats;
}

std::string XAIDataAggregator::getHealthStatus() {
    return computeHealthStatus();
}

// ========== Statistics ==========

XAIDataAggregator::AggregatorStats XAIDataAggregator::getStats() const {
    AggregatorStats stats;
    
    stats.events_ingested = events_ingested_.load();
    stats.threats_ingested = threats_ingested_.load();
    stats.events_streamed = events_streamed_.load();
    stats.callbacks_triggered = callbacks_triggered_.load();
    
    {
        std::lock_guard<std::mutex> lock(event_queue_mutex_);
        stats.event_queue_size = event_queue_.size();
    }
    
    {
        std::lock_guard<std::mutex> lock(threat_queue_mutex_);
        stats.threat_queue_size = threat_queue_.size();
    }
    
    return stats;
}

// ========== Private Methods ==========

void XAIDataAggregator::streamingLoop() {
    while (streaming_active_.load()) {
        processEventQueue();
        processThreatQueue();
        
        // Sleep briefly
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void XAIDataAggregator::processEventQueue() {
    std::vector<telemetry::TelemetryEvent> events_to_process;
    
    // Get events from queue
    {
        std::lock_guard<std::mutex> lock(event_queue_mutex_);
        
        while (!event_queue_.empty()) {
            events_to_process.push_back(event_queue_.front());
            event_queue_.pop();
        }
    }
    
    // Process events
    for (const auto& event : events_to_process) {
        notifyEventCallbacks(event);
        events_streamed_.fetch_add(1);
    }
}

void XAIDataAggregator::processThreatQueue() {
    std::vector<ThreatInfo> threats_to_process;
    
    // Get threats from queue
    {
        std::lock_guard<std::mutex> lock(threat_queue_mutex_);
        
        while (!threat_queue_.empty()) {
            threats_to_process.push_back(threat_queue_.front());
            threat_queue_.pop();
        }
    }
    
    // Process threats
    for (const auto& threat : threats_to_process) {
        notifyThreatCallbacks(threat);
    }
}

void XAIDataAggregator::updateRecentEventsCache(const telemetry::TelemetryEvent& event) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    recent_events_cache_.insert(recent_events_cache_.begin(), event);
    
    // Keep only last 1000 events
    if (recent_events_cache_.size() > 1000) {
        recent_events_cache_.resize(1000);
    }
}

void XAIDataAggregator::updateRecentThreatsCache(const ThreatInfo& threat) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    recent_threats_cache_.insert(recent_threats_cache_.begin(), threat);
    
    // Keep only last 500 threats
    if (recent_threats_cache_.size() > 500) {
        recent_threats_cache_.resize(500);
    }
}

void XAIDataAggregator::notifyEventCallbacks(const telemetry::TelemetryEvent& event) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    
    for (const auto& callback : event_callbacks_) {
        try {
            callback(event);
            callbacks_triggered_.fetch_add(1);
        } catch (const std::exception& e) {
            std::cerr << "[XAI Aggregator] Event callback error: " << e.what() << std::endl;
        }
    }
}

void XAIDataAggregator::notifyThreatCallbacks(const ThreatInfo& threat) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    
    for (const auto& callback : threat_callbacks_) {
        try {
            callback(threat);
            callbacks_triggered_.fetch_add(1);
        } catch (const std::exception& e) {
            std::cerr << "[XAI Aggregator] Threat callback error: " << e.what() << std::endl;
        }
    }
}

XAIDataAggregator::ThreatMetrics XAIDataAggregator::computeTodayMetrics() {
    ThreatMetrics metrics;
    
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    auto now = std::chrono::system_clock::now();
    auto today_start = std::chrono::system_clock::from_time_t(
        std::chrono::system_clock::to_time_t(now) / 86400 * 86400
    );
    
    for (const auto& threat : recent_threats_cache_) {
        // Check if threat is from today
        // (In production, would parse threat.timestamp)
        
        metrics.total_threats++;
        
        // Count by type
        metrics.threats_by_type[threat.threat_type]++;
        
        // Count by severity
        metrics.threats_by_severity[threat.severity]++;
        
        // Count actions
        if (threat.recommended_action == "BLOCK" || 
            threat.recommended_action == "QUARANTINE") {
            metrics.blocked_threats++;
        }
        
        if (threat.recommended_action == "QUARANTINE") {
            metrics.quarantined_files++;
        }
        
        if (threat.recommended_action == "SANDBOX") {
            metrics.sandboxed_files++;
        }
    }
    
    return metrics;
}

std::string XAIDataAggregator::computeHealthStatus() {
    std::lock_guard<std::mutex> lock(status_mutex_);
    
    int running_count = 0;
    int error_count = 0;
    int total_count = component_status_.size();
    
    for (const auto& [component, status] : component_status_) {
        if (status == "RUNNING") {
            running_count++;
        } else if (status == "ERROR" || status == "STOPPED") {
            error_count++;
        }
    }
    
    if (error_count > 0) {
        return "CRITICAL";
    } else if (running_count < total_count) {
        return "DEGRADED";
    } else {
        return "HEALTHY";
    }
}

} // namespace xai
