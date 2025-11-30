// CrashReporter.hpp - Automatic crash reporting system
#pragma once

#include <string>
#include <cstdint>

namespace DIREWOLF {
namespace Monitoring {

struct CrashReport {
    struct SystemInfo {
        std::string os;
        std::string osVersion;
        std::string cpuArchitecture;
        uint64_t totalMemoryMB;
        uint64_t availableMemoryMB;
    };
    
    struct ApplicationInfo {
        std::string version;
        std::string buildDate;
        std::string buildTime;
    };
    
    std::string timestamp;
    int signal;
    std::string signalName;
    std::string stackTrace;
    SystemInfo systemInfo;
    ApplicationInfo applicationInfo;
};

class CrashReporter {
public:
    explicit CrashReporter(const std::string& reportUrl);
    ~CrashReporter();
    
    void handleCrash(int signal);
    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }
    
private:
    void installHandlers();
    std::string captureStackTrace();
    CrashReport::SystemInfo collectSystemInfo();
    CrashReport::ApplicationInfo collectApplicationInfo();
    void saveCrashReport(const CrashReport& report);
    void submitCrashReport(const CrashReport& report);
    
    static std::string getCurrentTimestamp();
    static std::string getSignalName(int signal);
    
#ifdef _WIN32
    static LONG WINAPI windowsExceptionHandler(EXCEPTION_POINTERS* exceptionInfo);
#endif
    
    std::string reportUrl_;
    bool enabled_;
};

} // namespace Monitoring
} // namespace DIREWOLF
