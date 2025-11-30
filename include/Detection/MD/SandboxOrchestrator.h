#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

// Forward declarations
class DRLFramework;

enum class SandboxType {
    POSITIVE,  // For initially detected malware
    NEGATIVE   // For false negative detection
};

enum class SandboxResult {
    CLEAN,
    MALWARE_DETECTED,
    ERROR
};

struct SandboxAnalysisResult {
    SandboxResult result;
    std::vector<std::string> attackPatterns;
    std::vector<std::string> behaviors;
    float threatScore;
    std::string detailedReport;
};

class SandboxOrchestrator {
public:
    SandboxOrchestrator(std::shared_ptr<DRLFramework> drlFramework);
    ~SandboxOrchestrator();
    
    // Run file in positive sandbox (for detected malware)
    SandboxAnalysisResult runInPositiveSandbox(const std::string& filePath);
    
    // Run file in negative sandbox (for false negative check)
    SandboxAnalysisResult runInNegativeSandbox(const std::string& filePath);
    
    // Clean or delete malicious file
    bool cleanOrDeleteFile(const std::string& filePath, const SandboxAnalysisResult& analysis);
    
    // Get cleaned data packet
    std::vector<uint8_t> getCleanedDataPacket(const std::string& filePath);
    
private:
    std::shared_ptr<DRLFramework> drlFramework_;
    std::string positiveSandboxPath_;
    std::string negativeSandboxPath_;
    
    // Execute file in isolated environment
    SandboxAnalysisResult executeInIsolatedEnvironment(
        const std::string& filePath,
        SandboxType sandboxType
    );
    
    // Observe file behavior
    std::vector<std::string> observeBehavior(const std::string& filePath, SandboxType sandboxType);
    
    // Extract attack patterns
    std::vector<std::string> extractAttackPatterns(const std::vector<std::string>& behaviors);
    
    // Clean malicious content
    bool performCleaning(const std::string& filePath);
};
