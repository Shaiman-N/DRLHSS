#include "Sandbox/Windows/WindowsSandbox.hpp"

#ifdef _WIN32
#include <iostream>
#include <sstream>
#include <random>
#include <chrono>
#include <psapi.h>
#include <tlhelp32.h>

namespace sandbox {

WindowsSandbox::WindowsSandbox(SandboxType type)
    : type_(type), initialized_(false), job_object_(nullptr),
      app_container_sid_(nullptr), child_process_(nullptr), child_process_id_(0) {
    // Generate unique sandbox ID
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999);
    sandbox_id_ = "sandbox_" + std::to_string(dis(gen));
}

WindowsSandbox::~WindowsSandbox() {
    cleanup();
}

bool WindowsSandbox::initialize(const SandboxConfig& config) {
    std::lock_guard<std::mutex> lock(sandbox_mutex_);
    
    if (initialized_) {
        return true;
    }
    
    config_ = config;
    if (config_.sandbox_id.empty()) {
        config_.sandbox_id = sandbox_id_;
    } else {
        sandbox_id_ = config_.sandbox_id;
    }
    
    // Setup Job Object
    if (!setupJobObject()) {
        std::cerr << "[WindowsSandbox] Failed to setup job object" << std::endl;
        return false;
    }
    
    // Setup AppContainer
    if (!setupAppContainer()) {
        std::cerr << "[WindowsSandbox] Failed to setup AppContainer" << std::endl;
        return false;
    }
    
    // Setup file system redirection
    if (!setupFileSystemRedirection()) {
        std::cerr << "[WindowsSandbox] Failed to setup file system redirection" << std::endl;
        return false;
    }
    
    // Setup registry virtualization
    if (!setupRegistryVirtualization()) {
        std::cerr << "[WindowsSandbox] Failed to setup registry virtualization" << std::endl;
        return false;
    }
    
    initialized_ = true;
    std::cout << "[WindowsSandbox] Initialized: " << sandbox_id_ << std::endl;
    return true;
}

SandboxResult WindowsSandbox::execute(const std::string& file_path,
                                      const std::vector<std::string>& args) {
    std::lock_guard<std::mutex> lock(sandbox_mutex_);
    
    SandboxResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (!initialized_) {
        std::cerr << "[WindowsSandbox] Not initialized" << std::endl;
        return result;
    }
    
    PROCESS_INFORMATION pi = {0};
    
    // Create sandboxed process
    if (!createSandboxedProcess(file_path, args, pi)) {
        std::cerr << "[WindowsSandbox] Failed to create sandboxed process" << std::endl;
        return result;
    }
    
    child_process_ = pi.hProcess;
    child_process_id_ = pi.dwProcessId;
    
    // Wait for completion
    if (!waitForCompletion(pi.hProcess, result)) {
        std::cerr << "[WindowsSandbox] Process execution failed or timed out" << std::endl;
        TerminateProcess(pi.hProcess, 1);
    }
    
    // Monitor behavior
    monitorBehavior(result);
    
    // Cleanup process handles
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    child_process_ = nullptr;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    result.success = true;
    return result;
}

SandboxResult WindowsSandbox::analyzePacket(const std::vector<uint8_t>& packet_data) {
    // Create temporary file for packet analysis
    std::wstring temp_file = stringToWString(virtual_fs_path_ + "\\packet.bin");
    
    HANDLE hFile = CreateFileW(temp_file.c_str(), GENERIC_WRITE, 0, nullptr,
                               CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    
    if (hFile != INVALID_HANDLE_VALUE) {
        DWORD written;
        WriteFile(hFile, packet_data.data(), packet_data.size(), &written, nullptr);
        CloseHandle(hFile);
    }
    
    return execute(wstringToString(temp_file), {});
}

void WindowsSandbox::cleanup() {
    std::lock_guard<std::mutex> lock(sandbox_mutex_);
    
    if (!initialized_) {
        return;
    }
    
    // Terminate any running processes
    if (child_process_) {
        TerminateProcess(child_process_, 1);
        CloseHandle(child_process_);
        child_process_ = nullptr;
    }
    
    // Cleanup job object
    cleanupJobObject();
    
    // Cleanup AppContainer
    deleteAppContainerProfile();
    
    // Cleanup virtual file system
    cleanupVirtualFileSystem();
    
    // Cleanup virtual registry
    cleanupVirtualRegistry();
    
    initialized_ = false;
    std::cout << "[WindowsSandbox] Cleaned up: " << sandbox_id_ << std::endl;
}

void WindowsSandbox::reset() {
    cleanup();
    initialize(config_);
}

bool WindowsSandbox::isReady() const {
    return initialized_;
}

std::string WindowsSandbox::getSandboxId() const {
    return sandbox_id_;
}

// Private methods

bool WindowsSandbox::setupJobObject() {
    // Create job object
    std::wstring job_name = stringToWString("Global\\" + sandbox_id_);
    job_object_ = CreateJobObjectW(nullptr, job_name.c_str());
    
    if (!job_object_) {
        return false;
    }
    
    // Set job limits
    return setJobLimits();
}

bool WindowsSandbox::setJobLimits() {
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits = {0};
    
    // Set memory limit
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
    limits.ProcessMemoryLimit = config_.memory_limit_mb * 1024 * 1024;
    
    // Set CPU limit
    limits.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_PROCESS_TIME;
    limits.BasicLimitInformation.PerProcessUserTimeLimit.QuadPart = 
        config_.timeout_seconds * 10000000LL; // 100-nanosecond intervals
    
    // Set process limit
    limits.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_ACTIVE_PROCESS;
    limits.BasicLimitInformation.ActiveProcessLimit = 10;
    
    if (!SetInformationJobObject(job_object_, JobObjectExtendedLimitInformation,
                                 &limits, sizeof(limits))) {
        return false;
    }
    
    // Kill on job close
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION kill_on_close = {0};
    kill_on_close.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    
    SetInformationJobObject(job_object_, JobObjectExtendedLimitInformation,
                           &kill_on_close, sizeof(kill_on_close));
    
    return true;
}

void WindowsSandbox::cleanupJobObject() {
    if (job_object_) {
        CloseHandle(job_object_);
        job_object_ = nullptr;
    }
}

bool WindowsSandbox::setupAppContainer() {
    app_container_name_ = stringToWString(sandbox_id_);
    
    return createAppContainerProfile();
}

bool WindowsSandbox::createAppContainerProfile() {
    HRESULT hr = CreateAppContainerProfile(
        app_container_name_.c_str(),
        app_container_name_.c_str(),
        L"Sandbox AppContainer",
        nullptr,
        0,
        &app_container_sid_
    );
    
    if (FAILED(hr) && hr != HRESULT_FROM_WIN32(ERROR_ALREADY_EXISTS)) {
        return false;
    }
    
    return true;
}

void WindowsSandbox::deleteAppContainerProfile() {
    if (!app_container_name_.empty()) {
        DeleteAppContainerProfile(app_container_name_.c_str());
    }
    
    if (app_container_sid_) {
        FreeSid(app_container_sid_);
        app_container_sid_ = nullptr;
    }
}

bool WindowsSandbox::setupFileSystemRedirection() {
    return createVirtualFileSystem();
}

bool WindowsSandbox::createVirtualFileSystem() {
    // Create virtual file system directory
    char temp_path[MAX_PATH];
    GetTempPathA(MAX_PATH, temp_path);
    
    virtual_fs_path_ = std::string(temp_path) + sandbox_id_;
    
    if (!CreateDirectoryA(virtual_fs_path_.c_str(), nullptr)) {
        if (GetLastError() != ERROR_ALREADY_EXISTS) {
            return false;
        }
    }
    
    return true;
}

void WindowsSandbox::cleanupVirtualFileSystem() {
    if (!virtual_fs_path_.empty()) {
        // Remove directory recursively
        std::string cmd = "rmdir /s /q \"" + virtual_fs_path_ + "\"";
        system(cmd.c_str());
    }
}

bool WindowsSandbox::setupRegistryVirtualization() {
    return createVirtualRegistry();
}

bool WindowsSandbox::createVirtualRegistry() {
    // Create virtual registry key
    std::string reg_path = "Software\\Sandbox\\" + sandbox_id_;
    virtual_registry_path_ = reg_path;
    
    HKEY hKey;
    LONG result = RegCreateKeyExA(HKEY_CURRENT_USER, reg_path.c_str(), 0, nullptr,
                                  REG_OPTION_VOLATILE, KEY_ALL_ACCESS, nullptr,
                                  &hKey, nullptr);
    
    if (result == ERROR_SUCCESS) {
        RegCloseKey(hKey);
        return true;
    }
    
    return false;
}

void WindowsSandbox::cleanupVirtualRegistry() {
    if (!virtual_registry_path_.empty()) {
        RegDeleteTreeA(HKEY_CURRENT_USER, virtual_registry_path_.c_str());
    }
}

bool WindowsSandbox::setupNetworkIsolation() {
    // Network isolation via Windows Filtering Platform would go here
    // This is complex and requires WFP API integration
    return true;
}

bool WindowsSandbox::setIntegrityLevel(const std::wstring& level) {
    // Set process integrity level (Low, Medium, High)
    // This requires token manipulation
    return true;
}

void WindowsSandbox::monitorBehavior(SandboxResult& result) {
    monitorFileSystem(result);
    monitorRegistry(result);
    monitorNetwork(result);
    monitorProcesses(result);
    calculateThreatScore(result);
}

void WindowsSandbox::monitorFileSystem(SandboxResult& result) {
    // Check for file modifications in virtual file system
    WIN32_FIND_DATAA findData;
    std::string search_path = virtual_fs_path_ + "\\*";
    HANDLE hFind = FindFirstFileA(search_path.c_str(), &findData);
    
    if (hFind != INVALID_HANDLE_VALUE) {
        int file_count = 0;
        do {
            if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                file_count++;
                result.accessed_files.push_back(findData.cFileName);
            }
        } while (FindNextFileA(hFind, &findData));
        
        FindClose(hFind);
        result.file_system_modified = (file_count > 0);
    }
}

void WindowsSandbox::monitorRegistry(SandboxResult& result) {
    // Check for registry modifications
    HKEY hKey;
    if (RegOpenKeyExA(HKEY_CURRENT_USER, virtual_registry_path_.c_str(), 0,
                     KEY_READ, &hKey) == ERROR_SUCCESS) {
        DWORD subkeys = 0;
        DWORD values = 0;
        
        RegQueryInfoKeyA(hKey, nullptr, nullptr, nullptr, &subkeys, nullptr,
                        nullptr, &values, nullptr, nullptr, nullptr, nullptr);
        
        result.registry_modified = (subkeys > 0 || values > 0);
        RegCloseKey(hKey);
    }
}

void WindowsSandbox::monitorNetwork(SandboxResult& result) {
    // Monitor network connections
    // This would use GetExtendedTcpTable or similar APIs
    result.network_activity_detected = false;
}

void WindowsSandbox::monitorProcesses(SandboxResult& result) {
    // Check for child processes
    HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    
    if (hSnapshot != INVALID_HANDLE_VALUE) {
        PROCESSENTRY32 pe32;
        pe32.dwSize = sizeof(PROCESSENTRY32);
        
        if (Process32First(hSnapshot, &pe32)) {
            int child_count = 0;
            do {
                if (pe32.th32ParentProcessID == child_process_id_) {
                    child_count++;
                }
            } while (Process32Next(hSnapshot, &pe32));
            
            result.process_created = (child_count > 0);
        }
        
        CloseHandle(hSnapshot);
    }
}

void WindowsSandbox::calculateThreatScore(SandboxResult& result) {
    int score = 0;
    
    if (result.file_system_modified) score += 20;
    if (result.registry_modified) score += 25;
    if (result.network_activity_detected) score += 25;
    if (result.process_created) score += 20;
    if (result.memory_injection_detected) score += 30;
    if (result.suspicious_api_calls) score += 25;
    
    result.threat_score = std::min(score, 100);
}

bool WindowsSandbox::createSandboxedProcess(const std::string& file_path,
                                            const std::vector<std::string>& args,
                                            PROCESS_INFORMATION& pi) {
    STARTUPINFOA si = {0};
    si.cb = sizeof(si);
    
    // Build command line
    std::string cmdline = file_path;
    for (const auto& arg : args) {
        cmdline += " " + arg;
    }
    
    // Create process suspended
    if (!CreateProcessA(nullptr, const_cast<char*>(cmdline.c_str()), nullptr, nullptr,
                       FALSE, CREATE_SUSPENDED, nullptr, virtual_fs_path_.c_str(),
                       &si, &pi)) {
        return false;
    }
    
    // Assign to job object
    if (!AssignProcessToJobObject(job_object_, pi.hProcess)) {
        TerminateProcess(pi.hProcess, 1);
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        return false;
    }
    
    // Resume process
    ResumeThread(pi.hThread);
    
    return true;
}

bool WindowsSandbox::waitForCompletion(HANDLE process, SandboxResult& result) {
    DWORD timeout_ms = config_.timeout_seconds * 1000;
    DWORD wait_result = WaitForSingleObject(process, timeout_ms);
    
    if (wait_result == WAIT_OBJECT_0) {
        // Process completed
        DWORD exit_code;
        if (GetExitCodeProcess(process, &exit_code)) {
            result.exit_code = exit_code;
            return true;
        }
    }
    
    // Timeout or error
    return false;
}

std::wstring WindowsSandbox::stringToWString(const std::string& str) {
    if (str.empty()) return std::wstring();
    
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), nullptr, 0);
    std::wstring wstr(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstr[0], size_needed);
    
    return wstr;
}

std::string WindowsSandbox::wstringToString(const std::wstring& wstr) {
    if (wstr.empty()) return std::string();
    
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(),
                                         nullptr, 0, nullptr, nullptr);
    std::string str(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &str[0], size_needed,
                       nullptr, nullptr);
    
    return str;
}

} // namespace sandbox

#else
// Non-Windows stub
namespace sandbox {
WindowsSandbox::WindowsSandbox(SandboxType type) : type_(type), initialized_(false) {}
WindowsSandbox::~WindowsSandbox() {}
bool WindowsSandbox::initialize(const SandboxConfig&) { return false; }
SandboxResult WindowsSandbox::execute(const std::string&, const std::vector<std::string>&) { return SandboxResult(); }
SandboxResult WindowsSandbox::analyzePacket(const std::vector<uint8_t>&) { return SandboxResult(); }
void WindowsSandbox::cleanup() {}
void WindowsSandbox::reset() {}
bool WindowsSandbox::isReady() const { return false; }
std::string WindowsSandbox::getSandboxId() const { return ""; }
}
#endif
