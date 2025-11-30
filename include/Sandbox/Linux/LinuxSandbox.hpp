#pragma once

#include "../SandboxInterface.hpp"
#include <sys/types.h>
#include <unistd.h>
#include <mutex>

namespace sandbox {

/**
 * @brief Linux-specific sandbox implementation using namespaces and cgroups
 * 
 * Features:
 * - PID namespace isolation
 * - Network namespace isolation
 * - Mount namespace with overlay filesystem
 * - UTS namespace for hostname isolation
 * - IPC namespace for inter-process communication isolation
 * - cgroups for resource limits
 * - seccomp for syscall filtering
 */
class LinuxSandbox : public ISandbox {
public:
    LinuxSandbox(SandboxType type);
    ~LinuxSandbox() override;
    
    // ISandbox interface implementation
    bool initialize(const SandboxConfig& config) override;
    SandboxResult execute(const std::string& file_path,
                         const std::vector<std::string>& args = {}) override;
    SandboxResult analyzePacket(const std::vector<uint8_t>& packet_data) override;
    void cleanup() override;
    void reset() override;
    bool isReady() const override;
    std::string getSandboxId() const override;
    std::string getPlatform() const override { return "Linux"; }

private:
    // Namespace management
    bool setupNamespaces();
    bool setupPIDNamespace();
    bool setupNetworkNamespace();
    bool setupMountNamespace();
    bool setupUTSNamespace();
    bool setupIPCNamespace();
    
    // Filesystem management
    bool setupOverlayFilesystem();
    bool mountOverlay(const std::string& lower, const std::string& upper,
                     const std::string& work, const std::string& merged);
    void unmountOverlay();
    
    // Resource limits
    bool setupCgroups();
    bool setCPULimit(uint32_t percent);
    bool setMemoryLimit(uint64_t mb);
    void cleanupCgroups();
    
    // Security
    bool setupSeccomp();
    bool dropPrivileges();
    
    // Monitoring
    void monitorBehavior(SandboxResult& result);
    void monitorFileSystem(SandboxResult& result);
    void monitorNetwork(SandboxResult& result);
    void monitorProcesses(SandboxResult& result);
    void calculateThreatScore(SandboxResult& result);
    
    // Execution
    pid_t forkAndExecute(const std::string& file_path,
                        const std::vector<std::string>& args);
    bool waitForCompletion(pid_t pid, SandboxResult& result);
    
    SandboxType type_;
    SandboxConfig config_;
    bool initialized_;
    std::string sandbox_id_;
    std::string overlay_upper_;
    std::string overlay_work_;
    std::string overlay_merged_;
    std::string cgroup_path_;
    pid_t child_pid_;
    std::mutex sandbox_mutex_;
};

} // namespace sandbox
