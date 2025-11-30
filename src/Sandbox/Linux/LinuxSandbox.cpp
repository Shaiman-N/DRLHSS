#include "Sandbox/Linux/LinuxSandbox.hpp"
#include <sys/mount.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sched.h>
#include <signal.h>
#include <fcntl.h>
#include <unistd.h>
#include <seccomp.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <chrono>
#include <random>

namespace sandbox {

LinuxSandbox::LinuxSandbox(SandboxType type)
    : type_(type), initialized_(false), child_pid_(-1) {
    // Generate unique sandbox ID
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999);
    sandbox_id_ = "sandbox_" + std::to_string(dis(gen));
}

LinuxSandbox::~LinuxSandbox() {
    cleanup();
}

bool LinuxSandbox::initialize(const SandboxConfig& config) {
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
    
    // Setup overlay filesystem paths
    overlay_upper_ = "/tmp/" + sandbox_id_ + "/upper";
    overlay_work_ = "/tmp/" + sandbox_id_ + "/work";
    overlay_merged_ = "/tmp/" + sandbox_id_ + "/merged";
    
    // Create directories
    system(("mkdir -p " + overlay_upper_).c_str());
    system(("mkdir -p " + overlay_work_).c_str());
    system(("mkdir -p " + overlay_merged_).c_str());
    
    // Setup overlay filesystem
    if (!setupOverlayFilesystem()) {
        std::cerr << "[LinuxSandbox] Failed to setup overlay filesystem" << std::endl;
        return false;
    }
    
    // Setup cgroups for resource limits
    if (!setupCgroups()) {
        std::cerr << "[LinuxSandbox] Failed to setup cgroups" << std::endl;
        return false;
    }
    
    initialized_ = true;
    std::cout << "[LinuxSandbox] Initialized: " << sandbox_id_ << std::endl;
    return true;
}

SandboxResult LinuxSandbox::execute(const std::string& file_path,
                                    const std::vector<std::string>& args) {
    std::lock_guard<std::mutex> lock(sandbox_mutex_);
    
    SandboxResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (!initialized_) {
        std::cerr << "[LinuxSandbox] Not initialized" << std::endl;
        return result;
    }
    
    // Fork and execute in isolated environment
    child_pid_ = forkAndExecute(file_path, args);
    
    if (child_pid_ < 0) {
        std::cerr << "[LinuxSandbox] Fork failed" << std::endl;
        return result;
    }
    
    // Wait for completion with timeout
    if (!waitForCompletion(child_pid_, result)) {
        std::cerr << "[LinuxSandbox] Execution failed or timed out" << std::endl;
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

SandboxResult LinuxSandbox::analyzePacket(const std::vector<uint8_t>& packet_data) {
    // For packet analysis, we create a temporary file and analyze it
    std::string temp_file = "/tmp/" + sandbox_id_ + "/packet.bin";
    
    std::ofstream out(temp_file, std::ios::binary);
    out.write(reinterpret_cast<const char*>(packet_data.data()), packet_data.size());
    out.close();
    
    // Execute analysis tool on the packet
    return execute(temp_file, {});
}

void LinuxSandbox::cleanup() {
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
    
    // Cleanup cgroups
    cleanupCgroups();
    
    // Unmount overlay
    unmountOverlay();
    
    // Remove temporary directories
    system(("rm -rf /tmp/" + sandbox_id_).c_str());
    
    initialized_ = false;
    std::cout << "[LinuxSandbox] Cleaned up: " << sandbox_id_ << std::endl;
}

void LinuxSandbox::reset() {
    cleanup();
    initialize(config_);
}

bool LinuxSandbox::isReady() const {
    return initialized_;
}

std::string LinuxSandbox::getSandboxId() const {
    return sandbox_id_;
}

// Private methods

bool LinuxSandbox::setupNamespaces() {
    // Create new namespaces
    int flags = CLONE_NEWPID | CLONE_NEWNET | CLONE_NEWNS | 
                CLONE_NEWUTS | CLONE_NEWIPC;
    
    if (unshare(flags) != 0) {
        std::cerr << "[LinuxSandbox] Failed to create namespaces: " 
                  << strerror(errno) << std::endl;
        return false;
    }
    
    return true;
}

bool LinuxSandbox::setupOverlayFilesystem() {
    std::string lower = config_.base_image_path.empty() ? "/" : config_.base_image_path;
    
    return mountOverlay(lower, overlay_upper_, overlay_work_, overlay_merged_);
}

bool LinuxSandbox::mountOverlay(const std::string& lower, const std::string& upper,
                                const std::string& work, const std::string& merged) {
    std::string options = "lowerdir=" + lower + ",upperdir=" + upper + 
                         ",workdir=" + work;
    
    if (mount("overlay", merged.c_str(), "overlay", 0, options.c_str()) != 0) {
        std::cerr << "[LinuxSandbox] Failed to mount overlay: " 
                  << strerror(errno) << std::endl;
        return false;
    }
    
    return true;
}

void LinuxSandbox::unmountOverlay() {
    if (!overlay_merged_.empty()) {
        umount2(overlay_merged_.c_str(), MNT_DETACH);
    }
}

bool LinuxSandbox::setupCgroups() {
    cgroup_path_ = "/sys/fs/cgroup/" + sandbox_id_;
    
    // Create cgroup directory
    if (mkdir(cgroup_path_.c_str(), 0755) != 0 && errno != EEXIST) {
        std::cerr << "[LinuxSandbox] Failed to create cgroup: " 
                  << strerror(errno) << std::endl;
        return false;
    }
    
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

bool LinuxSandbox::setCPULimit(uint32_t percent) {
    std::string cpu_quota_file = cgroup_path_ + "/cpu.max";
    std::ofstream out(cpu_quota_file);
    
    if (!out.is_open()) {
        return false;
    }
    
    // Set CPU quota (percent of 100000 microseconds)
    int quota = (percent * 100000) / 100;
    out << quota << " 100000" << std::endl;
    out.close();
    
    return true;
}

bool LinuxSandbox::setMemoryLimit(uint64_t mb) {
    std::string memory_limit_file = cgroup_path_ + "/memory.max";
    std::ofstream out(memory_limit_file);
    
    if (!out.is_open()) {
        return false;
    }
    
    out << (mb * 1024 * 1024) << std::endl;
    out.close();
    
    return true;
}

void LinuxSandbox::cleanupCgroups() {
    if (!cgroup_path_.empty()) {
        rmdir(cgroup_path_.c_str());
    }
}

bool LinuxSandbox::setupSeccomp() {
    scmp_filter_ctx ctx = seccomp_init(SCMP_ACT_ALLOW);
    
    if (ctx == nullptr) {
        return false;
    }
    
    // Block dangerous syscalls
    seccomp_rule_add(ctx, SCMP_ACT_KILL, SCMP_SYS(ptrace), 0);
    seccomp_rule_add(ctx, SCMP_ACT_KILL, SCMP_SYS(reboot), 0);
    seccomp_rule_add(ctx, SCMP_ACT_KILL, SCMP_SYS(kexec_load), 0);
    
    if (seccomp_load(ctx) != 0) {
        seccomp_release(ctx);
        return false;
    }
    
    seccomp_release(ctx);
    return true;
}

bool LinuxSandbox::dropPrivileges() {
    // Drop to nobody user
    if (setuid(65534) != 0) {
        return false;
    }
    
    if (setgid(65534) != 0) {
        return false;
    }
    
    return true;
}

void LinuxSandbox::monitorBehavior(SandboxResult& result) {
    monitorFileSystem(result);
    monitorNetwork(result);
    monitorProcesses(result);
    calculateThreatScore(result);
}

void LinuxSandbox::monitorFileSystem(SandboxResult& result) {
    // Check for file modifications in overlay upper directory
    std::string cmd = "find " + overlay_upper_ + " -type f 2>/dev/null | wc -l";
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

void LinuxSandbox::monitorNetwork(SandboxResult& result) {
    // Check for network connections
    std::string netstat_file = "/proc/net/tcp";
    std::ifstream in(netstat_file);
    
    if (in.is_open()) {
        std::string line;
        int connection_count = 0;
        
        while (std::getline(in, line)) {
            connection_count++;
        }
        
        result.network_activity_detected = (connection_count > 1);
        in.close();
    }
}

void LinuxSandbox::monitorProcesses(SandboxResult& result) {
    // Check for child processes
    std::string cmd = "ps --ppid " + std::to_string(child_pid_) + " 2>/dev/null | wc -l";
    FILE* pipe = popen(cmd.c_str(), "r");
    
    if (pipe) {
        char buffer[128];
        if (fgets(buffer, sizeof(buffer), pipe)) {
            int process_count = std::atoi(buffer);
            result.process_created = (process_count > 1);
        }
        pclose(pipe);
    }
}

void LinuxSandbox::calculateThreatScore(SandboxResult& result) {
    int score = 0;
    
    if (result.file_system_modified) score += 20;
    if (result.registry_modified) score += 15;
    if (result.network_activity_detected) score += 25;
    if (result.process_created) score += 20;
    if (result.memory_injection_detected) score += 30;
    if (result.suspicious_api_calls) score += 25;
    
    result.threat_score = std::min(score, 100);
}

pid_t LinuxSandbox::forkAndExecute(const std::string& file_path,
                                   const std::vector<std::string>& args) {
    pid_t pid = fork();
    
    if (pid == 0) {
        // Child process
        
        // Setup namespaces
        setupNamespaces();
        
        // Change root to overlay merged directory
        if (chroot(overlay_merged_.c_str()) != 0) {
            std::cerr << "[LinuxSandbox] chroot failed" << std::endl;
            exit(1);
        }
        
        chdir("/");
        
        // Setup seccomp
        setupSeccomp();
        
        // Drop privileges
        dropPrivileges();
        
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
        std::cerr << "[LinuxSandbox] exec failed: " << strerror(errno) << std::endl;
        exit(1);
    }
    
    return pid;
}

bool LinuxSandbox::waitForCompletion(pid_t pid, SandboxResult& result) {
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

} // namespace sandbox
