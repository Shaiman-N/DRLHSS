#pragma once

#include "../SandboxInterface.hpp"

#ifdef __APPLE__
#include <sys/types.h>
#include <unistd.h>
#include <sandbox.h>
#endif

#include <mutex>

namespace sandbox {

/**
 * @brief macOS-specific sandbox implementation
 * 
 * Features:
 * - macOS Sandbox profiles (SBPL - Sandbox Profile Language)
 * - TCC (Transparency, Consent, and Control) integration
 * - File system quarantine
 * - Code signing verification
 * - Entitlements-based restrictions
 * - XPC service isolation
 */
class MacOSSandbox : public ISandbox {
public:
    MacOSSandbox(SandboxType type);
    ~MacOSSandbox() override;
    
    // ISandbox interface implementation
    bool initialize(const SandboxConfig& config) override;
    SandboxResult execute(const std::string& file_path,
                         const std::vector<std::string>& args = {}) override;
    SandboxResult analyzePacket(const std::vector<uint8_t>& packet_data) override;
    void cleanup() override;
    void reset() override;
    bool isReady() const override;
    std::string getSandboxId() const override;
    std::string getPlatform() const override { return "macOS"; }

private:
#ifdef __APPLE__
    // Sandbox profile management
    bool setupSandboxProfile();
    std::string generateSandboxProfile();
    bool applySandboxProfile(const std::string& profile);
    
    // File system quarantine
    bool setupFileQuarantine();
    bool setQuarantineAttribute(const std::string& path);
    void cleanupQuarantine();
    
    // Code signing
    bool verifyCodeSignature(const std::string& file_path);
    bool checkEntitlements(const std::string& file_path);
    
    // TCC (Transparency, Consent, and Control)
    bool setupTCCRestrictions();
    bool requestTCCPermissions();
    
    // Resource limits
    bool setupResourceLimits();
    bool setCPULimit(uint32_t percent);
    bool setMemoryLimit(uint64_t mb);
    
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
    
    // Helper functions
    bool createSandboxDirectory();
    void cleanupSandboxDirectory();
    
    std::string sandbox_profile_;
    std::string sandbox_directory_;
    pid_t child_pid_;
#endif
    
    SandboxType type_;
    SandboxConfig config_;
    bool initialized_;
    std::string sandbox_id_;
    std::mutex sandbox_mutex_;
};

} // namespace sandbox
