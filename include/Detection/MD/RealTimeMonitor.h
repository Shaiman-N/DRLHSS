#ifndef REALTIMEMONITOR_H
#define REALTIMEMONITOR_H

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <functional>
#include <windows.h>

/**
 * Real-Time Monitoring System
 * Monitors persistence mechanisms, file system, and registry for malware activity
 */
class RealTimeMonitor {
public:
    RealTimeMonitor();
    ~RealTimeMonitor();
    
    // Start/Stop monitoring
    bool startMonitoring();
    void stopMonitoring();
    bool isMonitoring() const { return monitoring; }
    
    // Set callback for detected threats
    void setThreatCallback(std::function<void(const std::string&, const std::string&)> callback);
    
    // Configuration
    void enableRegistryMonitoring(bool enable) { monitorRegistry = enable; }
    void enableFileSystemMonitoring(bool enable) { monitorFileSystem = enable; }
    void enableStartupMonitoring(bool enable) { monitorStartup = enable; }
    void enableNetworkMonitoring(bool enable) { monitorNetwork = enable; }
    
    // Manual scans
    void scanPersistenceLocations();
    void scanStartupFolders();
    void scanRegistryAutorun();
    
    // Statistics
    struct MonitorStats {
        int filesScanned;
        int threatsDetected;
        int registryChanges;
        int startupChanges;
        std::string lastThreat;
        std::string startTime;
    };
    
    MonitorStats getStats() const { return stats; }

private:
    // Monitoring threads
    void registryMonitorThread();
    void fileSystemMonitorThread();
    void startupMonitorThread();
    void networkMonitorThread();
    
    // Detection methods
    bool checkRegistryKey(const std::string& keyPath);
    bool checkFile(const std::string& filePath);
    bool checkStartupEntry(const std::string& name, const std::string& path);
    
    // Persistence locations
    std::vector<std::string> getAutorunRegistryKeys();
    std::vector<std::string> getStartupFolders();
    
    // Threat handling
    void onThreatDetected(const std::string& location, const std::string& description);
    
    // State
    std::atomic<bool> monitoring;
    std::atomic<bool> monitorRegistry;
    std::atomic<bool> monitorFileSystem;
    std::atomic<bool> monitorStartup;
    std::atomic<bool> monitorNetwork;
    
    // Threads
    std::thread registryThread;
    std::thread fileSystemThread;
    std::thread startupThread;
    std::thread networkThread;
    
    // Callback
    std::function<void(const std::string&, const std::string&)> threatCallback;
    
    // Statistics
    MonitorStats stats;
    
    // Windows handles
    HANDLE registryNotifyHandle;
    HANDLE fileSystemNotifyHandle;
};

#endif // REALTIMEMONITOR_H
