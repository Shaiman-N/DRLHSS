#include "RealTimeMonitor.h"
#include <iostream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>

RealTimeMonitor::RealTimeMonitor() 
    : monitoring(false),
      monitorRegistry(true),
      monitorFileSystem(true),
      monitorStartup(true),
      monitorNetwork(false),
      registryNotifyHandle(nullptr),
      fileSystemNotifyHandle(nullptr) {
    
    stats.filesScanned = 0;
    stats.threatsDetected = 0;
    stats.registryChanges = 0;
    stats.startupChanges = 0;
    
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    stats.startTime = ss.str();
}

RealTimeMonitor::~RealTimeMonitor() {
    stopMonitoring();
}

bool RealTimeMonitor::startMonitoring() {
    if (monitoring) {
        std::cout << "[RealTimeMonitor] Already monitoring" << std::endl;
        return false;
    }
    
    std::cout << "[RealTimeMonitor] Starting real-time protection..." << std::endl;
    monitoring = true;
    
    // Start monitoring threads
    if (monitorRegistry) {
        registryThread = std::thread(&RealTimeMonitor::registryMonitorThread, this);
        std::cout << "[RealTimeMonitor] Registry monitoring started" << std::endl;
    }
    
    if (monitorFileSystem) {
        fileSystemThread = std::thread(&RealTimeMonitor::fileSystemMonitorThread, this);
        std::cout << "[RealTimeMonitor] File system monitoring started" << std::endl;
    }
    
    if (monitorStartup) {
        startupThread = std::thread(&RealTimeMonitor::startupMonitorThread, this);
        std::cout << "[RealTimeMonitor] Startup monitoring started" << std::endl;
    }
    
    if (monitorNetwork) {
        networkThread = std::thread(&RealTimeMonitor::networkMonitorThread, this);
        std::cout << "[RealTimeMonitor] Network monitoring started" << std::endl;
    }
    
    // Initial scan
    std::cout << "[RealTimeMonitor] Performing initial scan..." << std::endl;
    scanPersistenceLocations();
    
    std::cout << "[RealTimeMonitor] Real-time protection active" << std::endl;
    return true;
}

void RealTimeMonitor::stopMonitoring() {
    if (!monitoring) {
        return;
    }
    
    std::cout << "[RealTimeMonitor] Stopping monitoring..." << std::endl;
    monitoring = false;
    
    // Wait for threads to finish
    if (registryThread.joinable()) registryThread.join();
    if (fileSystemThread.joinable()) fileSystemThread.join();
    if (startupThread.joinable()) startupThread.join();
    if (networkThread.joinable()) networkThread.join();
    
    std::cout << "[RealTimeMonitor] Monitoring stopped" << std::endl;
}

void RealTimeMonitor::setThreatCallback(std::function<void(const std::string&, const std::string&)> callback) {
    threatCallback = callback;
}

std::vector<std::string> RealTimeMonitor::getAutorunRegistryKeys() {
    return {
        "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run",
        "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\RunOnce",
        "SOFTWARE\\WOW6432Node\\Microsoft\\Windows\\CurrentVersion\\Run",
        "SOFTWARE\\WOW6432Node\\Microsoft\\Windows\\CurrentVersion\\RunOnce",
        "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Shell Folders",
        "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Explorer\\User Shell Folders"
    };
}

std::vector<std::string> RealTimeMonitor::getStartupFolders() {
    std::vector<std::string> folders;
    
    // Get user startup folder
    char userStartup[MAX_PATH];
    if (SHGetFolderPathA(NULL, CSIDL_STARTUP, NULL, 0, userStartup) == S_OK) {
        folders.push_back(userStartup);
    }
    
    // Get common startup folder
    char commonStartup[MAX_PATH];
    if (SHGetFolderPathA(NULL, CSIDL_COMMON_STARTUP, NULL, 0, commonStartup) == S_OK) {
        folders.push_back(commonStartup);
    }
    
    return folders;
}

void RealTimeMonitor::registryMonitorThread() {
    std::cout << "[RegistryMonitor] Thread started" << std::endl;
    
    while (monitoring) {
        // Get autorun registry keys
        auto keys = getAutorunRegistryKeys();
        
        for (const auto& keyPath : keys) {
            HKEY hKey;
            std::string fullPath = "HKEY_LOCAL_MACHINE\\" + keyPath;
            
            // Open registry key
            if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, keyPath.c_str(), 0, KEY_READ | KEY_NOTIFY, &hKey) == ERROR_SUCCESS) {
                
                // Enumerate values
                DWORD index = 0;
                char valueName[256];
                DWORD valueNameSize = sizeof(valueName);
                BYTE valueData[1024];
                DWORD valueDataSize = sizeof(valueData);
                DWORD valueType;
                
                while (RegEnumValueA(hKey, index++, valueName, &valueNameSize, NULL, &valueType, valueData, &valueDataSize) == ERROR_SUCCESS) {
                    
                    if (valueType == REG_SZ || valueType == REG_EXPAND_SZ) {
                        std::string value(reinterpret_cast<char*>(valueData));
                        
                        // Check if this is a suspicious entry
                        if (checkRegistryKey(value)) {
                            std::string threat = "Suspicious autorun entry: " + std::string(valueName) + " -> " + value;
                            onThreatDetected(fullPath, threat);
                        }
                    }
                    
                    valueNameSize = sizeof(valueName);
                    valueDataSize = sizeof(valueData);
                }
                
                RegCloseKey(hKey);
            }
        }
        
        // Sleep for 5 seconds before next check
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
    
    std::cout << "[RegistryMonitor] Thread stopped" << std::endl;
}

void RealTimeMonitor::fileSystemMonitorThread() {
    std::cout << "[FileSystemMonitor] Thread started" << std::endl;
    
    // Monitor common malware locations
    std::vector<std::string> monitorPaths = {
        "C:\\Windows\\System32",
        "C:\\Windows\\Temp",
        "C:\\Users\\Public"
    };
    
    while (monitoring) {
        for (const auto& path : monitorPaths) {
            WIN32_FIND_DATAA findData;
            HANDLE hFind = FindFirstFileA((path + "\\*").c_str(), &findData);
            
            if (hFind != INVALID_HANDLE_VALUE) {
                do {
                    if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                        std::string filePath = path + "\\" + findData.cFileName;
                        
                        // Check recently modified files
                        FILETIME currentTime;
                        GetSystemTimeAsFileTime(&currentTime);
                        
                        ULARGE_INTEGER fileTime, curTime;
                        fileTime.LowPart = findData.ftLastWriteTime.dwLowDateTime;
                        fileTime.HighPart = findData.ftLastWriteTime.dwHighDateTime;
                        curTime.LowPart = currentTime.dwLowDateTime;
                        curTime.HighPart = currentTime.dwHighDateTime;
                        
                        // If modified in last 60 seconds
                        if ((curTime.QuadPart - fileTime.QuadPart) < 600000000ULL) {
                            if (checkFile(filePath)) {
                                onThreatDetected(filePath, "Suspicious file activity detected");
                            }
                            stats.filesScanned++;
                        }
                    }
                } while (FindNextFileA(hFind, &findData));
                
                FindClose(hFind);
            }
        }
        
        // Sleep for 10 seconds
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }
    
    std::cout << "[FileSystemMonitor] Thread stopped" << std::endl;
}

void RealTimeMonitor::startupMonitorThread() {
    std::cout << "[StartupMonitor] Thread started" << std::endl;
    
    while (monitoring) {
        auto folders = getStartupFolders();
        
        for (const auto& folder : folders) {
            WIN32_FIND_DATAA findData;
            HANDLE hFind = FindFirstFileA((folder + "\\*").c_str(), &findData);
            
            if (hFind != INVALID_HANDLE_VALUE) {
                do {
                    if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                        std::string fileName = findData.cFileName;
                        std::string filePath = folder + "\\" + fileName;
                        
                        if (checkStartupEntry(fileName, filePath)) {
                            onThreatDetected(folder, "Suspicious startup entry: " + fileName);
                            stats.startupChanges++;
                        }
                    }
                } while (FindNextFileA(hFind, &findData));
                
                FindClose(hFind);
            }
        }
        
        // Sleep for 15 seconds
        std::this_thread::sleep_for(std::chrono::seconds(15));
    }
    
    std::cout << "[StartupMonitor] Thread stopped" << std::endl;
}

void RealTimeMonitor::networkMonitorThread() {
    std::cout << "[NetworkMonitor] Thread started" << std::endl;
    
    while (monitoring) {
        // Network monitoring would go here
        // This is a placeholder for future implementation
        
        std::this_thread::sleep_for(std::chrono::seconds(30));
    }
    
    std::cout << "[NetworkMonitor] Thread stopped" << std::endl;
}

bool RealTimeMonitor::checkRegistryKey(const std::string& value) {
    // Check for suspicious patterns in registry values
    std::vector<std::string> suspiciousPatterns = {
        "temp\\",
        "appdata\\local\\temp",
        ".tmp.exe",
        "powershell",
        "cmd.exe /c",
        "wscript",
        "cscript"
    };
    
    std::string lowerValue = value;
    std::transform(lowerValue.begin(), lowerValue.end(), lowerValue.begin(), ::tolower);
    
    for (const auto& pattern : suspiciousPatterns) {
        if (lowerValue.find(pattern) != std::string::npos) {
            return true;
        }
    }
    
    return false;
}

bool RealTimeMonitor::checkFile(const std::string& filePath) {
    // Check for suspicious file characteristics
    std::string lowerPath = filePath;
    std::transform(lowerPath.begin(), lowerPath.end(), lowerPath.begin(), ::tolower);
    
    // Suspicious extensions
    std::vector<std::string> suspiciousExt = {
        ".exe", ".dll", ".scr", ".bat", ".cmd", ".vbs", ".js"
    };
    
    for (const auto& ext : suspiciousExt) {
        if (lowerPath.find(ext) != std::string::npos) {
            // Could trigger full malware scan here
            return true;
        }
    }
    
    return false;
}

bool RealTimeMonitor::checkStartupEntry(const std::string& name, const std::string& path) {
    // Check if startup entry is suspicious
    std::string lowerName = name;
    std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
    
    // Suspicious names
    std::vector<std::string> suspiciousNames = {
        "update", "svchost", "system", "windows", "microsoft"
    };
    
    for (const auto& suspicious : suspiciousNames) {
        if (lowerName.find(suspicious) != std::string::npos) {
            return true;
        }
    }
    
    return false;
}

void RealTimeMonitor::onThreatDetected(const std::string& location, const std::string& description) {
    stats.threatsDetected++;
    stats.lastThreat = description;
    
    std::cout << "[THREAT DETECTED] " << location << std::endl;
    std::cout << "  Description: " << description << std::endl;
    
    if (threatCallback) {
        threatCallback(location, description);
    }
}

void RealTimeMonitor::scanPersistenceLocations() {
    std::cout << "[RealTimeMonitor] Scanning persistence locations..." << std::endl;
    
    scanRegistryAutorun();
    scanStartupFolders();
    
    std::cout << "[RealTimeMonitor] Persistence scan complete" << std::endl;
}

void RealTimeMonitor::scanStartupFolders() {
    std::cout << "[RealTimeMonitor] Scanning startup folders..." << std::endl;
    
    auto folders = getStartupFolders();
    int itemsFound = 0;
    
    for (const auto& folder : folders) {
        std::cout << "  Checking: " << folder << std::endl;
        
        WIN32_FIND_DATAA findData;
        HANDLE hFind = FindFirstFileA((folder + "\\*").c_str(), &findData);
        
        if (hFind != INVALID_HANDLE_VALUE) {
            do {
                if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                    itemsFound++;
                    std::cout << "    - " << findData.cFileName << std::endl;
                }
            } while (FindNextFileA(hFind, &findData));
            
            FindClose(hFind);
        }
    }
    
    std::cout << "  Total startup items: " << itemsFound << std::endl;
}

void RealTimeMonitor::scanRegistryAutorun() {
    std::cout << "[RealTimeMonitor] Scanning registry autorun keys..." << std::endl;
    
    auto keys = getAutorunRegistryKeys();
    int entriesFound = 0;
    
    for (const auto& keyPath : keys) {
        HKEY hKey;
        
        if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, keyPath.c_str(), 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
            std::cout << "  Checking: HKLM\\" << keyPath << std::endl;
            
            DWORD index = 0;
            char valueName[256];
            DWORD valueNameSize = sizeof(valueName);
            
            while (RegEnumValueA(hKey, index++, valueName, &valueNameSize, NULL, NULL, NULL, NULL) == ERROR_SUCCESS) {
                entriesFound++;
                valueNameSize = sizeof(valueName);
            }
            
            RegCloseKey(hKey);
        }
    }
    
    std::cout << "  Total autorun entries: " << entriesFound << std::endl;
    stats.registryChanges = entriesFound;
}
