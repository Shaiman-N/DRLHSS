#pragma once

#include "Telemetry/TelemetryEvent.hpp"
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <functional>
#include <condition_variable>

namespace telemetry {

/**
 * @brief Host Telemetry Collector
 * 
 * Collects real-time telemetry from the host system including:
 * - Process creation/termination
 * - File system operations
 * - Registry/config changes
 * - Network connections
 * - System calls
 * 
 * This is the primary telemetry source for the detection layer.
 */
class HostTelemetryCollector {
public:
    using TelemetryCallback = std::function<void(const TelemetryEvent&)>;
    
    struct CollectorConfig {
        bool enable_process_monitoring;
        bool enable_file_monitoring;
        bool enable_registry_monitoring;
        bool enable_network_monitoring;
        bool enable_syscall_monitoring;
        int max_queue_size;
        int collection_interval_ms;
        
        CollectorConfig() :
            enable_process_monitoring(true),
            enable_file_monitoring(true),
            enable_registry_monitoring(true),
            enable_network_monitoring(true),
            enable_syscall_monitoring(false),  // Requires elevated privileges
            max_queue_size(10000),
            collection_interval_ms(100) {}
    };
    
    explicit HostTelemetryCollector(const CollectorConfig& config);
    ~HostTelemetryCollector();
    
    // Start/Stop collection
    bool start();
    void stop();
    bool isRunning() const { return running_.load(); }
    
    // Set callback for telemetry events
    void setCallback(TelemetryCallback callback);
    
    // Get collected events (polling mode)
    std::vector<TelemetryEvent> getEvents(int max_count = 100);
    
    // Statistics
    struct CollectorStats {
        uint64_t total_events_collected;
        uint64_t process_events;
        uint64_t file_events;
        uint64_t registry_events;
        uint64_t network_events;
        uint64_t syscall_events;
        uint64_t queue_size;
        uint64_t dropped_events;
    };
    
    CollectorStats getStatistics() const;
    
private:
    // Collection threads
    void processMonitorThread();
    void fileMonitorThread();
    void registryMonitorThread();
    void networkMonitorThread();
    void syscallMonitorThread();
    
    // Platform-specific implementations
    void collectProcessEvents();
    void collectFileEvents();
    void collectRegistryEvents();
    void collectNetworkEvents();
    void collectSyscallEvents();
    
    // Event queue management
    void enqueueEvent(const TelemetryEvent& event);
    
    CollectorConfig config_;
    std::atomic<bool> running_;
    
    // Collection threads
    std::thread process_thread_;
    std::thread file_thread_;
    std::thread registry_thread_;
    std::thread network_thread_;
    std::thread syscall_thread_;
    
    // Event queue
    std::queue<TelemetryEvent> event_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // Callback
    TelemetryCallback callback_;
    std::mutex callback_mutex_;
    
    // Statistics
    mutable std::atomic<uint64_t> total_events_{0};
    mutable std::atomic<uint64_t> process_events_{0};
    mutable std::atomic<uint64_t> file_events_{0};
    mutable std::atomic<uint64_t> registry_events_{0};
    mutable std::atomic<uint64_t> network_events_{0};
    mutable std::atomic<uint64_t> syscall_events_{0};
    mutable std::atomic<uint64_t> dropped_events_{0};
};

} // namespace telemetry
