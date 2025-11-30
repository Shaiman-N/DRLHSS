/**
 * @file XAIDataAggregator.hpp
 * @brief XAI Data Aggregator for DIREWOLF
 * 
 * Aggregates data from all DRLHSS components (Telemetry, AV, NIDPS, DRL, Sandbox)
 * and provides unified access for explainability and permission requests.
 */

#pragma once

#include "XAITypes.hpp"
#include "Telemetry/TelemetryEvent.hpp"
#include "DRL/TelemetryData.hpp"
#include "DRL/AttackPattern.hpp"
#include "DB/DatabaseManager.hpp"
#include <memory>
#include <vector>
#include <string>
#include <mutex>
#include <queue>
#include <functional>
#include <atomic>
#include <thread>

namespace xai {

/**
 * @brief Real-time event stream callback
 */
using EventStreamCallback = std::function<void(const telemetry::TelemetryEvent&)>;

/**
 * @brief Threat detection callback
 */
using ThreatDetectionCallback = std::function<void(const ThreatInfo&)>;

/**
 * @brief System state snapshot
 */
struct SystemSnapshot {
    // Current state
    int threats_today = 0;
    int active_alerts = 0;
    std::string health_status = "HEALTHY";
    float drl_confidence = 0.0f;
    
    // Component status
    std::map<std::string, std::string> component_status;
    
    // Recent events (last 100)
    std::vector<telemetry::TelemetryEvent> recent_events;
    
    // Recent threats (last 50)
    std::vector<ThreatInfo> recent_threats;
    
    // Performance metrics
    struct {
        uint64_t events_processed = 0;
        uint64_t threats_detected = 0;
        uint64_t threats_blocked = 0;
        double avg_detection_time_ms = 0.0;
    } metrics;
    
    std::chrono::system_clock::time_point snapshot_time;
};

/**
 * @brief XAI Data Aggregator
 * 
 * Central hub for collecting and aggregating data from all DRLHSS components.
 * Provides real-time event streaming, batch data retrieval, and system state queries.
 */
class XAIDataAggregator {
public:
    /**
     * @brief Constructor
     * @param db_path Path to DRLHSS database
     */
    explicit XAIDataAggregator(const std::string& db_path);
    
    /**
     * @brief Destructor
     */
    ~XAIDataAggregator();
    
    /**
     * @brief Initialize aggregator
     * @return True if successful
     */
    bool initialize();
    
    /**
     * @brief Start real-time event streaming
     */
    void startStreaming();
    
    /**
     * @brief Stop real-time event streaming
     */
    void stopStreaming();
    
    // ========== Real-Time Event Streaming ==========
    
    /**
     * @brief Register callback for real-time events
     * @param callback Function to call on each event
     */
    void registerEventCallback(EventStreamCallback callback);
    
    /**
     * @brief Register callback for threat detections
     * @param callback Function to call on each threat
     */
    void registerThreatCallback(ThreatDetectionCallback callback);
    
    /**
     * @brief Ingest telemetry event (called by telemetry system)
     * @param event Telemetry event
     */
    void ingestEvent(const telemetry::TelemetryEvent& event);
    
    /**
     * @brief Ingest threat detection (called by detection systems)
     * @param threat Threat information
     */
    void ingestThreat(const ThreatInfo& threat);
    
    // ========== Batch Data Retrieval ==========
    
    /**
     * @brief Get recent telemetry events
     * @param limit Maximum number of events
     * @param event_type Filter by event type (optional)
     * @return Vector of events
     */
    std::vector<telemetry::TelemetryEvent> getRecentEvents(
        int limit = 100,
        telemetry::EventType event_type = telemetry::EventType::UNKNOWN
    );
    
    /**
     * @brief Get events by time range
     * @param start_time Start timestamp
     * @param end_time End timestamp
     * @return Vector of events
     */
    std::vector<telemetry::TelemetryEvent> getEventsByTimeRange(
        std::chrono::system_clock::time_point start_time,
        std::chrono::system_clock::time_point end_time
    );
    
    /**
     * @brief Get events by process
     * @param pid Process ID
     * @param limit Maximum number of events
     * @return Vector of events
     */
    std::vector<telemetry::TelemetryEvent> getEventsByProcess(int pid, int limit = 100);
    
    /**
     * @brief Get recent threats
     * @param limit Maximum number of threats
     * @return Vector of threats
     */
    std::vector<ThreatInfo> getRecentThreats(int limit = 50);
    
    /**
     * @brief Get threats by severity
     * @param severity Minimum severity level
     * @param limit Maximum number of threats
     * @return Vector of threats
     */
    std::vector<ThreatInfo> getThreatsBySeverity(ThreatSeverity severity, int limit = 50);
    
    /**
     * @brief Get attack patterns
     * @param attack_type Filter by attack type (optional)
     * @param limit Maximum number of patterns
     * @return Vector of attack patterns
     */
    std::vector<drl::AttackPattern> getAttackPatterns(
        const std::string& attack_type = "",
        int limit = 100
    );
    
    // ========== System State Queries ==========
    
    /**
     * @brief Get current system state snapshot
     * @return System snapshot
     */
    SystemSnapshot getSystemSnapshot();
    
    /**
     * @brief Get component status
     * @param component_name Component name (AV, NIDPS, DRL, Sandbox, etc.)
     * @return Status string (RUNNING, STOPPED, ERROR, etc.)
     */
    std::string getComponentStatus(const std::string& component_name);
    
    /**
     * @brief Update component status
     * @param component_name Component name
     * @param status Status string
     */
    void updateComponentStatus(const std::string& component_name, const std::string& status);
    
    /**
     * @brief Get threat metrics for today
     * @return Threat metrics
     */
    struct ThreatMetrics {
        int total_threats = 0;
        int blocked_threats = 0;
        int quarantined_files = 0;
        int sandboxed_files = 0;
        int false_positives = 0;
        std::map<std::string, int> threats_by_type;
        std::map<ThreatSeverity, int> threats_by_severity;
    };
    
    ThreatMetrics getTodayMetrics();
    
    /**
     * @brief Get DRL agent confidence
     * @return Confidence score (0.0 - 1.0)
     */
    float getDRLConfidence();
    
    /**
     * @brief Update DRL confidence
     * @param confidence New confidence score
     */
    void updateDRLConfidence(float confidence);
    
    // ========== Threat Metrics Aggregation ==========
    
    /**
     * @brief Aggregate threat statistics
     * @param time_window Time window in hours
     * @return Aggregated statistics
     */
    struct AggregatedStats {
        int total_events = 0;
        int total_threats = 0;
        float detection_rate = 0.0f;
        float false_positive_rate = 0.0f;
        std::map<std::string, int> top_threat_types;
        std::map<std::string, int> top_processes;
        std::vector<std::pair<std::string, int>> timeline; // Hour -> count
    };
    
    AggregatedStats aggregateStats(int time_window_hours = 24);
    
    /**
     * @brief Get system health status
     * @return Health status (HEALTHY, DEGRADED, CRITICAL)
     */
    std::string getHealthStatus();
    
    // ========== Statistics ==========
    
    struct AggregatorStats {
        uint64_t events_ingested = 0;
        uint64_t threats_ingested = 0;
        uint64_t events_streamed = 0;
        uint64_t callbacks_triggered = 0;
        size_t event_queue_size = 0;
        size_t threat_queue_size = 0;
    };
    
    AggregatorStats getStats() const;

private:
    // Database
    std::unique_ptr<db::DatabaseManager> db_manager_;
    std::string db_path_;
    
    // Real-time streaming
    std::atomic<bool> streaming_active_{false};
    std::unique_ptr<std::thread> streaming_thread_;
    
    // Event queues
    std::queue<telemetry::TelemetryEvent> event_queue_;
    std::queue<ThreatInfo> threat_queue_;
    mutable std::mutex event_queue_mutex_;
    mutable std::mutex threat_queue_mutex_;
    
    // Callbacks
    std::vector<EventStreamCallback> event_callbacks_;
    std::vector<ThreatDetectionCallback> threat_callbacks_;
    mutable std::mutex callback_mutex_;
    
    // System state
    std::map<std::string, std::string> component_status_;
    mutable std::mutex status_mutex_;
    
    float drl_confidence_ = 0.0f;
    mutable std::mutex confidence_mutex_;
    
    // In-memory caches
    std::vector<telemetry::TelemetryEvent> recent_events_cache_;
    std::vector<ThreatInfo> recent_threats_cache_;
    mutable std::mutex cache_mutex_;
    
    // Statistics
    mutable std::atomic<uint64_t> events_ingested_{0};
    mutable std::atomic<uint64_t> threats_ingested_{0};
    mutable std::atomic<uint64_t> events_streamed_{0};
    mutable std::atomic<uint64_t> callbacks_triggered_{0};
    
    // Thread functions
    void streamingLoop();
    void processEventQueue();
    void processThreatQueue();
    
    // Helper methods
    void updateRecentEventsCache(const telemetry::TelemetryEvent& event);
    void updateRecentThreatsCache(const ThreatInfo& threat);
    void notifyEventCallbacks(const telemetry::TelemetryEvent& event);
    void notifyThreatCallbacks(const ThreatInfo& threat);
    
    // Metrics computation
    ThreatMetrics computeTodayMetrics();
    std::string computeHealthStatus();
};

} // namespace xai
