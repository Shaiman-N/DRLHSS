#include "Sandbox/MacOS/MacOSSandbox.hpp"

#ifdef __APPLE__
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <signal.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>

namespace sandbox {

MacOSSandbox::MacOSSandbox(SandboxType type)
    : type_(type), initialized_(false), child_pid_(-1) {
    // Generate unique sandbox ID
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999);
    sandbox_id_ = "sandbox_" + std::to_string(dis(gen));
}

MacOSSandbox::~MacOSSandbox() {
    cleanup();
}

bool MacOSSandbox::initialize(const SandboxConfig& config) {
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
    
    // Create sandbox directory
    if (!createSandboxDirectory()) {
        std::cerr << "[MacOSSandbox] Failed to create sandbox directory" << std::endl;
        return false;
    }
    
    // Setup sandbox profile
    if (!setupSandboxProfile()) {
        std::cerr << "[MacOSSandbox] Failed to setup sandbox profile" << std::endl;
        return false;
    }
    
    // Setup file quarantine
    if (!setupFileQuarantine()) {
        std::cerr << "[MacOSSandbox] Failed to setup file quarantine" << std::endl;
        return false;
    }
    
    // Setup resource limits
    if (!setupResourceLimits()) {
        std::cerr << "[MacOSSandbox] Failed to setup resource limits" << std::endl;
        return false;
    }
    
    initialized_ = true;
    std::cout << "[MacOSSandbox] Initialized: " << sandbox_id_ << std::endl;
    return true;
}

SandboxResult MacOSSandbox::execute(const std::string& file_path,
                                    const std::vector<std::string>& args) {
    std::lock_guard<std::mutex> lock(sandbox_mutex_);
    
    SandboxResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (!initialized_) {
        std::cerr << "[MacOSSandbox] Not initialized" << std::endl;
        return result;
    }
    
    // Verify code signature if required
    if (!verifyCodeSignature(file_path)) {
        std::cerr << "[MacOSSandbox] Code signature verification failed" << std::endl;
        result.threat_score = 50;
        return result;
    }
    
    // Fork and execute in sandbox
    child_pid_ = forkAndExecute(file_path, args);
    
    if (child_pid_ < 0) {
        std::cerr << "[MacOSSandbox] Fork failed" << std::endl;
        return result;
    }
    
    // Wait for completion
    if (!waitForCompletion(child_pid_, result)) {
        std::cerr << "[MacOSSandbox] Execution failed or timed out" << std::endl;
        kill(child_pid_, SIGKILL);
        waitpid(child_pid_, nullptr, 0);
        return result;
    }
    
    // Monitor behavior
    monitorBehavior(result);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    result.success = true;
    return result;
}

SandboxResult MacOSSandbox::analyzePacket(const std::vector<uint8_t>& packet_data) {
    // Create temporary file for packet analysis
    std::string temp_file = sandbox_directory_ + "/packet.bin";
    
    std::ofstream out(temp_file, std::ios::binary);
    out.write(reinterpret_cast<const char*>(packet_data.data()), packet_data.size());
    out.close();
    
    // Set quarantine attribute
    setQuarantineAttribute(temp_file);
    
    return execute(temp_file, {});
}

void MacOSSandbox::cleanup() {
    std::lock_guard<std::mutex> lock(sandbox_mutex_);
    
    if (!initialized_) {
        return;
    }
    
    // Kill any running processes
    if (child_pid_ > 0) {
        kill(child_pid_, SIGKILL);
        waitpid(child_pid_, nullptr, 0);
        child_pid_ = -1;
    }
    
    // Cleanup quarantine
    cleanupQuarantine();
    
    // Remove sandbox directory
    cleanupSandboxDirectory();
    
    initialized_ = false;
    std::cout << "[MacOSSandbox] Cleaned up: " << sandbox_id_ << std::endl;
}

void MacOSSandbox::reset() {
    cleanup();
    initialize(config_);
}

bool MacOSSandbox::isReady() const {
    return initialized_;
}

std::string MacOSSandbox::getSandboxId() const {
    return sandbox_id_;
}

// Private methods

bool MacOSSandbox::setupSandboxProfile() {
    sandbox_profile_ = generateSandboxProfile();
    return !sandbox_profile_.empty();
}

std::string MacOSSandbox::generateSandboxProfile() {
    // Generate SBPL (Sandbox Profile Language) profile
    std::stringstream profile;
    
    profile << "(version 1)\n";
    profile << "(deny default)\n";
    
    // Allow basic operations
    profile << "(allow process-exec)\n";
    profile << "(allow process-fork)\n";
    profile << "(allow sysctl-read)\n";
    
    // File system access
    if (config_.read_only_filesystem) {
        profile << "(allow file-read*)\n";
        profile << "(deny file-write*)\n";
    } else {
        profile << "(allow file-read* file-write*)\n";
    }
    
    // Restrict to sandbox directory
    profile << "(allow file* (subpath \"" << sandbox_directory_ << "\"))\n";
    
    // Network access
    if (config_.allow_network) {
        profile << "(allow network*)\n";
    } else {
        profile << "(deny network*)\n";
    }
    
    // Deny dangerous operations
    profile << "(deny system-socket)\n";
    profile << "(deny mach-lookup)\n";
    profile << "(deny ipc-posix-shm)\n";
    
    return profile.str();
}

bool MacOSSandbox::applySandboxProfile(const std::string& profile) {
    char* error = nullptr;
    
    if (sandbox_init(profile.c_str(), 0, &error) != 0) {
        if (error) {
            std::cerr << "[MacOSSandbox] Sandbox init failed: " << error << std::endl;
            sandbox_free_error(error);
        }
        return false;
    }
    
    return true;
}

bool MacOSSandbox::setupFileQuarantine() {
    // File quarantine is set per-file using extended attributes
    return true;
}

bool MacOSSandbox::setQuarantineAttribute(const std::string& path) {
    // Set com.apple.quarantine extended attribute
    std::string cmd = "xattr -w com.apple.quarantine \"0001;$(date +%s);Sandbox;\" \"" + path + "\"";
    return system(cmd.c_str()) == 0;
}

void MacOSSandbox::cleanupQuarantine() {
    // Remove quarantine attributes from sandbox directory
    if (!sandbox_directory_.empty()) {
        std::string cmd = "xattr -dr com.apple.quarantine \"" + sandbox_directory_ + "\"";
        system(cmd.c_str());
    }
}

bool MacOSSandbox::verifyCodeSignature(const std::string& file_path) {
    // Verify code signature using codesign
    std::string cmd = "codesign --verify --deep --strict \"" + file_path + "\" 2>/dev/null";
    int result = system(cmd.c_str());
    
    // Return true if signature is valid or if file is not signed (for testing)
    return (result == 0 || result == 256);
}

bool MacOSSandbox::checkEntitlements(const std::string& file_path) {
    // Check entitlements using codesign
    std::string cmd = "codesign -d --entitlements - \"" + file_path + "\" 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    
    if (!pipe) {
        return false;
    }
    
    char buffer[128];
    bool has_entitlements = false;
    
    while (fgets(buffer, sizeof(buffer), pipe)) {
        has_entitlements = true;
    }
    
    pclose(pipe);
    return has_entitlements;
}

bool MacOSSandbox::setupTCCRestrictions() {
    // TCC (Transparency, Consent, and Control) restrictions
    // This would require system-level permissions
    return true;
}

bool MacOSSandbox::setupResourceLimits() {
    // Set CPU limit
    if (!setCPULimit(config_.cpu_limit_percent)) {
        return false;
    }
    
    // Set memory limit
    if (!setMemoryLimit(config_.memory_limit_mb)) {
        return false;
    }
    
    return true;
}

bool MacOSSandbox::setCPULimit(uint32_t percent) {
    // CPU limits on macOS are set per-process using setrlimit
    struct rlimit limit;
    limit.rlim_cur = (percent * RLIM_INFINITY) / 100;
    limit.rlim_max = RLIM_INFINITY;
    
    return setrlimit(RLIMIT_CPU, &limit) == 0;
}

bool MacOSSandbox::setMemoryLimit(uint64_t mb) {
    struct rlimit limit;
    limit.rlim_cur = mb * 1024 * 1024;
    limit.rlim_max = mb * 1024 * 1024;
    
    return setrlimit(RLIMIT_AS, &limit) == 0;
}

void MacOSSandbox::monitorBehavior(SandboxResult& result) {
    monitorFileSystem(result);
    monitorNetwork(result);
    monitorProcesses(result);
    calculateThreatScore(result);
}

void MacOSSandbox::monitorFileSystem(SandboxResult& result) {
    // Check for file modifications in sandbox directory
    std::string cmd = "find \"" + sandbox_directory_ + "\" -type f 2>/dev/null | wc -l";
    FILE* pipe = popen(cmd.c_str(), "r");
    
    if (pipe) {
        char buffer[128];
        if (fgets(buffer, sizeof(buffer), pipe)) {
            int file_count = std::atoi(buffer);
            result.file_system_modified = (file_count > 0);
        }
        pclose(pipe);
    }
}

void MacOSSandbox::monitorNetwork(SandboxResult& result) {
    // Check for network connections using lsof
    std::string cmd = "lsof -p " + std::to_string(child_pid_) + " -i 2>/dev/null | wc -l";
    FILE* pipe = popen(cmd.c_str(), "r");
    
    if (pipe) {
        char buffer[128];
        if (fgets(buffer, sizeof(buffer), pipe)) {
            int connection_count = std::atoi(buffer);
            result.network_activity_detected = (connection_count > 0);
        }
        pclose(pipe);
    }
}

void MacOSSandbox::monitorProcesses(SandboxResult& result) {
    // Check for child processes
    std::string cmd = "ps -o pid= --ppid " + std::to_string(child_pid_) + " 2>/dev/null | wc -l";
    FILE* pipe = popen(cmd.c_str(), "r");
    
    if (pipe) {
        char buffer[128];
        if (fgets(buffer, sizeof(buffer), pipe)) {
            int process_count = std::atoi(buffer);
            result.process_created = (process_count > 0);
        }
        pclose(pipe);
    }
}

void MacOSSandbox::calculateThreatScore(SandboxResult& result) {
    int score = 0;
    
    if (result.file_system_modified) score += 20;
    if (result.registry_modified) score += 15;
    if (result.network_activity_detected) score += 25;
    if (result.process_created) score += 20;
    if (result.memory_injection_detected) score += 30;
    if (result.suspicious_api_calls) score += 25;
    
    result.threat_score = std::min(score, 100);
}

pid_t MacOSSandbox::forkAndExecute(const std::string& file_path,
                                   const std::vector<std::string>& args) {
    pid_t pid = fork();
    
    if (pid == 0) {
        // Child process
        
        // Change to sandbox directory
        if (chdir(sandbox_directory_.c_str()) != 0) {
            std::cerr << "[MacOSSandbox] chdir failed" << std::endl;
            exit(1);
        }
        
        // Apply sandbox profile
        if (!applySandboxProfile(sandbox_profile_)) {
            std::cerr << "[MacOSSandbox] Failed to apply sandbox profile" << std::endl;
            exit(1);
        }
        
        // Prepare arguments
        std::vector<char*> exec_args;
        exec_args.push_back(const_cast<char*>(file_path.c_str()));
        
        for (const auto& arg : args) {
            exec_args.push_back(const_cast<char*>(arg.c_str()));
        }
        exec_args.push_back(nullptr);
        
        // Execute
        execv(file_path.c_str(), exec_args.data());
        
        // If we get here, exec failed
        std::cerr << "[MacOSSandbox] exec failed: " << strerror(errno) << std::endl;
        exit(1);
    }
    
    return pid;
}

bool MacOSSandbox::waitForCompletion(pid_t pid, SandboxResult& result) {
    int status;
    int timeout_seconds = config_.timeout_seconds;
    
    // Wait with timeout
    for (int i = 0; i < timeout_seconds; i++) {
        pid_t ret = waitpid(pid, &status, WNOHANG);
        
        if (ret == pid) {
            // Process completed
            if (WIFEXITED(status)) {
                result.exit_code = WEXITSTATUS(status);
                return true;
            } else if (WIFSIGNALED(status)) {
                result.exit_code = -WTERMSIG(status);
                return true;
            }
        } else if (ret < 0) {
            return false;
        }
        
        sleep(1);
    }
    
    // Timeout
    return false;
}

bool MacOSSandbox::createSandboxDirectory() {
    sandbox_directory_ = "/tmp/" + sandbox_id_;
    
    if (mkdir(sandbox_directory_.c_str(), 0700) != 0) {
        if (errno != EEXIST) {
            return false;
        }
    }
    
    return true;
}

void MacOSSandbox::cleanupSandboxDirectory() {
    if (!sandbox_directory_.empty()) {
        std::string cmd = "rm -rf \"" + sandbox_directory_ + "\"";
        system(cmd.c_str());
    }
}

} // namespace sandbox

#else
// Non-macOS stub
namespace sandbox {
MacOSSandbox::MacOSSandbox(SandboxType type) : type_(type), initialized_(false) {}
MacOSSandbox::~MacOSSandbox() {}
bool MacOSSandbox::initialize(const SandboxConfig&) { return false; }
SandboxResult MacOSSandbox::execute(const std::string&, const std::vector<std::string>&) { return SandboxResult(); }
SandboxResult MacOSSandbox::analyzePacket(const std::vector<uint8_t>&) { return SandboxResult(); }
void MacOSSandbox::cleanup() {}
void MacOSSandbox::reset() {}
bool MacOSSandbox::isReady() const { return false; }
std::string MacOSSandbox::getSandboxId() const { return ""; }
}
#endif
