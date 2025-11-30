#pragma once

#include "../SandboxInterface.hpp"

#ifdef _WIN32
#include <windows.h>
#include <userenv.h>
#include <sddl.h>
#endif

#include <mutex>

namespace sandbox {

/**
 * @brief Windows-specific sandbox implementation
 * 
 * Features:
 * - Job Objects for process isolation and resource limits
 * - AppContainer for security isolation
 * - File system redirection
 * - Registry virtualization
 * - Network isolation via Windows Filtering Platform
 * - Integrity levels (Low/Medium/High)
 */
class WindowsSandbox : public ISandbox {
public:
    WindowsSandbox(SandboxType type);
    ~WindowsSandbox() override;
    
    // ISandbox interface implementation
    bool initialize(const SandboxConfig& config) override;
    SandboxResult execute(const std::string& file_path,
                         const std::vector<std::string>& args = {}) override;
    SandboxResult analyzePacket(const std::vector<uint8_t>& packet_data) override;
    void cleanup() override;
    void reset() override;
    bool isReady() const override;
    std::string getSandboxId() const override;
    std::string getPlatform() const override { return "Windows"; }

private:
#ifdef _WIN32
    // Job Object management
    bool setupJobObject();
    bool setJobLimits();
    void cleanupJobObject();
    
    // AppContainer management
    bool setupAppContainer();
    bool createAppContainerProfile();
    void deleteAppContainerProfile();
    
    // File system redirection
    bool setupFileSystemRedirection();
    bool createVirtualFileSystem();
    void cleanupVirtualFileSystem();
    
    // Registry virtualization
    bool setupRegistryVirtualization();
    bool createVirtualRegistry();
    void cleanupVirtualRegistry();
    
    // Network isolation
    bool setupNetworkIsolation();
    bool configureFirewallRules();
    void cleanupFirewallRules();
    
    // Security
    bool setIntegrityLevel(const std::wstring& level);
    bool restrictTokenPrivileges(HANDLE token);
    bool setupSecurityDescriptor();
    
    // Monitoring
    void monitorBehavior(SandboxResult& result);
    void monitorFileSystem(SandboxResult& result);
    void monitorRegistry(SandboxResult& result);
    void monitorNetwork(SandboxResult& result);
    void monitorProcesses(SandboxResult& result);
    void calculateThreatScore(SandboxResult& result);
    
    // Execution
    bool createSandboxedProcess(const std::string& file_path,
                               const std::vector<std::string>& args,
                               PROCESS_INFORMATION& pi);
    bool waitForCompletion(HANDLE process, SandboxResult& result);
    
    // Helper functions
    std::wstring stringToWString(const std::string& str);
    std::string wstringToString(const std::wstring& wstr);
    
    HANDLE job_object_;
    PSID app_container_sid_;
    std::wstring app_container_name_;
    std::string virtual_fs_path_;
    std::string virtual_registry_path_;
    HANDLE child_process_;
    DWORD child_process_id_;
#endif
    
    SandboxType type_;
    SandboxConfig config_;
    bool initialized_;
    std::string sandbox_id_;
    std::mutex sandbox_mutex_;
};

} // namespace sandbox
