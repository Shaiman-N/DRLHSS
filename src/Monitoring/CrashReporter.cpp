// CrashReporter.cpp - Automatic crash reporting system
#include "CrashReporter.hpp"
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <curl/curl.h>
#include <json/json.h>

#ifdef _WIN32
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
#elif defined(__APPLE__)
#include <execinfo.h>
#include <signal.h>
#elif defined(__linux__)
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
#endif

namespace DIREWOLF {
namespace Monitoring {

// Global crash reporter instance
static CrashReporter* g_crashReporter = nullptr;

// Signal handler for crashes
void crashSignalHandler(int signal) {
    if (g_crashReporter) {
        g_crashReporter->handleCrash(signal);
    }
    
    // Re-raise signal for default handling
    std::raise(signal);
}

CrashReporter::CrashReporter(const std::string& reportUrl)
    : reportUrl_(reportUrl)
    , enabled_(true)
{
    g_crashReporter = this;
    installHandlers();
}

CrashReporter::~CrashReporter() {
    g_crashReporter = nullptr;
}

void CrashReporter::installHandlers() {
#ifdef _WIN32
    // Windows: Use SetUnhandledExceptionFilter
    SetUnhandledExceptionFilter(windowsExceptionHandler);
#else
    // Unix: Install signal handlers
    signal(SIGSEGV, crashSignalHandler);  // Segmentation fault
    signal(SIGABRT, crashSignalHandler);  // Abort
    signal(SIGFPE, crashSignalHandler);   // Floating point exception
    signal(SIGILL, crashSignalHandler);   // Illegal instruction
#endif
}

void CrashReporter::handleCrash(int signal) {
    if (!enabled_) {
        return;
    }
    
    CrashReport report;
    report.timestamp = getCurrentTimestamp();
    report.signal = signal;
    report.signalName = getSignalName(signal);
    report.stackTrace = captureStackTrace();
    report.systemInfo = collectSystemInfo();
    report.applicationInfo = collectApplicationInfo();
    
    // Save crash report locally
    saveCrashReport(report);
    
    // Submit to server (async)
    std::thread([this, report]() {
        submitCrashReport(report);
    }).detach();
}

std::string CrashReporter::captureStackTrace() {
    std::stringstream ss;
    
#ifdef _WIN32
    // Windows stack trace
    void* stack[100];
    HANDLE process = GetCurrentProcess();
    SymInitialize(process, NULL, TRUE);
    
    WORD frames = CaptureStackBackTrace(0, 100, stack, NULL);
    
    SYMBOL_INFO* symbol = (SYMBOL_INFO*)calloc(sizeof(SYMBOL_INFO) + 256, 1);
    symbol->MaxNameLen = 255;
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
    
    for (WORD i = 0; i < frames; i++) {
        SymFromAddr(process, (DWORD64)(stack[i]), 0, symbol);
        ss << i << ": " << symbol->Name << " - 0x" << std::hex << symbol->Address << std::dec << "\n";
    }
    
    free(symbol);
    SymCleanup(process);
    
#else
    // Unix stack trace
    void* array[100];
    size_t size = backtrace(array, 100);
    char** strings = backtrace_symbols(array, size);
    
    for (size_t i = 0; i < size; i++) {
        ss << i << ": " << strings[i] << "\n";
    }
    
    free(strings);
#endif
    
    return ss.str();
}

CrashReport::SystemInfo CrashReporter::collectSystemInfo() {
    CrashReport::SystemInfo info;
    
#ifdef _WIN32
    info.os = "Windows";
    OSVERSIONINFOEX osvi;
    ZeroMemory(&osvi, sizeof(OSVERSIONINFOEX));
    osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);
    GetVersionEx((OSVERSIONINFO*)&osvi);
    info.osVersion = std::to_string(osvi.dwMajorVersion) + "." + std::to_string(osvi.dwMinorVersion);
    
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    info.cpuArchitecture = (sysInfo.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_AMD64) ? "x64" : "x86";
    
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    info.totalMemoryMB = memInfo.ullTotalPhys / (1024 * 1024);
    info.availableMemoryMB = memInfo.ullAvailPhys / (1024 * 1024);
    
#elif defined(__APPLE__)
    info.os = "macOS";
    // Get macOS version
    FILE* fp = popen("sw_vers -productVersion", "r");
    if (fp) {
        char buffer[128];
        if (fgets(buffer, sizeof(buffer), fp)) {
            info.osVersion = buffer;
        }
        pclose(fp);
    }
    info.cpuArchitecture = "x64";
    
#elif defined(__linux__)
    info.os = "Linux";
    // Get Linux version
    FILE* fp = popen("uname -r", "r");
    if (fp) {
        char buffer[128];
        if (fgets(buffer, sizeof(buffer), fp)) {
            info.osVersion = buffer;
        }
        pclose(fp);
    }
    info.cpuArchitecture = "x64";
#endif
    
    return info;
}

CrashReport::ApplicationInfo CrashReporter::collectApplicationInfo() {
    CrashReport::ApplicationInfo info;
    info.version = "1.0.0";  // Get from version header
    info.buildDate = __DATE__;
    info.buildTime = __TIME__;
    
    return info;
}

void CrashReporter::saveCrashReport(const CrashReport& report) {
    try {
        // Create crash reports directory
        std::string crashDir = "crashes/";
        
        // Generate filename with timestamp
        std::string filename = crashDir + "crash_" + report.timestamp + ".json";
        
        // Convert to JSON
        Json::Value root;
        root["timestamp"] = report.timestamp;
        root["signal"] = report.signal;
        root["signal_name"] = report.signalName;
        root["stack_trace"] = report.stackTrace;
        
        Json::Value sysInfo;
        sysInfo["os"] = report.systemInfo.os;
        sysInfo["os_version"] = report.systemInfo.osVersion;
        sysInfo["cpu_architecture"] = report.systemInfo.cpuArchitecture;
        sysInfo["total_memory_mb"] = report.systemInfo.totalMemoryMB;
        sysInfo["available_memory_mb"] = report.systemInfo.availableMemoryMB;
        root["system_info"] = sysInfo;
        
        Json::Value appInfo;
        appInfo["version"] = report.applicationInfo.version;
        appInfo["build_date"] = report.applicationInfo.buildDate;
        appInfo["build_time"] = report.applicationInfo.buildTime;
        root["application_info"] = appInfo;
        
        // Write to file
        std::ofstream file(filename);
        Json::StreamWriterBuilder writer;
        file << Json::writeString(writer, root);
        file.close();
        
    } catch (const std::exception& e) {
        // Failed to save crash report
        // Can't do much here
    }
}

void CrashReporter::submitCrashReport(const CrashReport& report) {
    if (reportUrl_.empty()) {
        return;
    }
    
    try {
        CURL* curl = curl_easy_init();
        if (!curl) {
            return;
        }
        
        // Convert to JSON
        Json::Value root;
        root["timestamp"] = report.timestamp;
        root["signal"] = report.signal;
        root["signal_name"] = report.signalName;
        root["stack_trace"] = report.stackTrace;
        
        Json::Value sysInfo;
        sysInfo["os"] = report.systemInfo.os;
        sysInfo["os_version"] = report.systemInfo.osVersion;
        sysInfo["cpu_architecture"] = report.systemInfo.cpuArchitecture;
        sysInfo["total_memory_mb"] = report.systemInfo.totalMemoryMB;
        sysInfo["available_memory_mb"] = report.systemInfo.availableMemoryMB;
        root["system_info"] = sysInfo;
        
        Json::Value appInfo;
        appInfo["version"] = report.applicationInfo.version;
        appInfo["build_date"] = report.applicationInfo.buildDate;
        appInfo["build_time"] = report.applicationInfo.buildTime;
        root["application_info"] = appInfo;
        
        Json::StreamWriterBuilder writer;
        std::string jsonStr = Json::writeString(writer, root);
        
        // Set up HTTP POST
        curl_easy_setopt(curl, CURLOPT_URL, reportUrl_.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonStr.c_str());
        
        struct curl_slist* headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        
        // Perform request
        CURLcode res = curl_easy_perform(curl);
        
        // Cleanup
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        
    } catch (const std::exception& e) {
        // Failed to submit crash report
    }
}

std::string CrashReporter::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S");
    return ss.str();
}

std::string CrashReporter::getSignalName(int signal) {
    switch (signal) {
        case SIGSEGV: return "SIGSEGV (Segmentation Fault)";
        case SIGABRT: return "SIGABRT (Abort)";
        case SIGFPE: return "SIGFPE (Floating Point Exception)";
        case SIGILL: return "SIGILL (Illegal Instruction)";
        default: return "Unknown Signal";
    }
}

#ifdef _WIN32
LONG WINAPI CrashReporter::windowsExceptionHandler(EXCEPTION_POINTERS* exceptionInfo) {
    if (g_crashReporter) {
        g_crashReporter->handleCrash(exceptionInfo->ExceptionRecord->ExceptionCode);
    }
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

} // namespace Monitoring
} // namespace DIREWOLF
