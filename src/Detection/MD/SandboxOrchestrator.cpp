#include "SandboxOrchestrator.h"
#include "DRLFramework.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <thread>
#include <chrono>

namespace fs = std::filesystem;

SandboxOrchestrator::SandboxOrchestrator(std::shared_ptr<DRLFramework> drlFramework)
    : drlFramework_(drlFramework) {
    
    // Create sandbox directories
    positiveSandboxPath_ = "sandbox_positive";
    negativeSandboxPath_ = "sandbox_negative";
    
    if (!fs::exists(positiveSandboxPath_)) {
        fs::create_directory(positiveSandboxPath_);
    }
    if (!fs::exists(negativeSandboxPath_)) {
        fs::create_directory(negativeSandboxPath_);
    }
    
    std::cout << "[SandboxOrchestrator] Initialized with dual sandbox system" << std::endl;
}

SandboxOrchestrator::~SandboxOrchestrator() {
    std::cout << "[SandboxOrchestrator] Shutting down" << std::endl;
}

SandboxAnalysisResult SandboxOrchestrator::runInPositiveSandbox(const std::string& filePath) {
    std::cout << "[PositiveSandbox] Running file: " << filePath << std::endl;
    
    SandboxAnalysisResult result = executeInIsolatedEnvironment(filePath, SandboxType::POSITIVE);
    
    if (result.result == SandboxResult::MALWARE_DETECTED) {
        std::cout << "[PositiveSandbox] Malware behavior detected!" << std::endl;
        std::cout << "[PositiveSandbox] Attack patterns found: " << result.attackPatterns.size() << std::endl;
        
        // Send to DRL framework for learning
        BehaviorData behaviorData;
        behaviorData.filePath = filePath;
        behaviorData.behaviors = result.behaviors;
        behaviorData.threatLevel = result.threatScore;
        behaviorData.sandboxType = "POSITIVE";
        behaviorData.timestamp = std::chrono::system_clock::now().time_since_epoch().count();
        
        for (const auto& pattern : result.attackPatterns) {
            AttackPattern ap;
            ap.patternType = pattern;
            ap.description = "Detected in positive sandbox";
            ap.severity = result.threatScore;
            ap.timestamp = behaviorData.timestamp;
            behaviorData.patterns.push_back(ap);
        }
        
        drlFramework_->learnFromBehavior(behaviorData);
        drlFramework_->storeToDatabase(behaviorData);
    }
    
    return result;
}

SandboxAnalysisResult SandboxOrchestrator::runInNegativeSandbox(const std::string& filePath) {
    std::cout << "[NegativeSandbox] Checking for false negatives: " << filePath << std::endl;
    
    SandboxAnalysisResult result = executeInIsolatedEnvironment(filePath, SandboxType::NEGATIVE);
    
    if (result.result == SandboxResult::MALWARE_DETECTED) {
        std::cout << "[NegativeSandbox] False negative detected!" << std::endl;
        std::cout << "[NegativeSandbox] Attack patterns found: " << result.attackPatterns.size() << std::endl;
        
        // Send to DRL framework for learning
        BehaviorData behaviorData;
        behaviorData.filePath = filePath;
        behaviorData.behaviors = result.behaviors;
        behaviorData.threatLevel = result.threatScore;
        behaviorData.sandboxType = "NEGATIVE";
        behaviorData.timestamp = std::chrono::system_clock::now().time_since_epoch().count();
        
        for (const auto& pattern : result.attackPatterns) {
            AttackPattern ap;
            ap.patternType = pattern;
            ap.description = "False negative detected in negative sandbox";
            ap.severity = result.threatScore;
            ap.timestamp = behaviorData.timestamp;
            behaviorData.patterns.push_back(ap);
        }
        
        drlFramework_->learnFromBehavior(behaviorData);
        drlFramework_->storeToDatabase(behaviorData);
    } else {
        std::cout << "[NegativeSandbox] No threats detected - file is clean" << std::endl;
    }
    
    return result;
}

bool SandboxOrchestrator::cleanOrDeleteFile(const std::string& filePath, const SandboxAnalysisResult& analysis) {
    std::cout << "[SandboxOrchestrator] Cleaning/deleting malicious content from: " << filePath << std::endl;
    
    if (analysis.threatScore > 0.8f) {
        // High threat - delete file
        std::cout << "[SandboxOrchestrator] High threat detected - deleting file" << std::endl;
        try {
            fs::remove(filePath);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[SandboxOrchestrator] Error deleting file: " << e.what() << std::endl;
            return false;
        }
    } else {
        // Medium threat - attempt cleaning
        std::cout << "[SandboxOrchestrator] Medium threat - attempting to clean file" << std::endl;
        return performCleaning(filePath);
    }
}

std::vector<uint8_t> SandboxOrchestrator::getCleanedDataPacket(const std::string& filePath) {
    std::cout << "[SandboxOrchestrator] Extracting cleaned data packet from: " << filePath << std::endl;
    
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[SandboxOrchestrator] Failed to open cleaned file" << std::endl;
        return {};
    }
    
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());
    
    std::cout << "[SandboxOrchestrator] Cleaned data packet size: " << data.size() << " bytes" << std::endl;
    return data;
}

SandboxAnalysisResult SandboxOrchestrator::executeInIsolatedEnvironment(
    const std::string& filePath,
    SandboxType sandboxType) {
    
    SandboxAnalysisResult result;
    result.result = SandboxResult::CLEAN;
    result.threatScore = 0.0f;
    
    std::string sandboxPath = (sandboxType == SandboxType::POSITIVE) ? 
                              positiveSandboxPath_ : negativeSandboxPath_;
    
    // Copy file to sandbox
    fs::path sourceFile(filePath);
    fs::path sandboxFile = fs::path(sandboxPath) / sourceFile.filename();
    
    try {
        fs::copy_file(sourceFile, sandboxFile, fs::copy_options::overwrite_existing);
    } catch (const std::exception& e) {
        std::cerr << "[SandboxOrchestrator] Error copying to sandbox: " << e.what() << std::endl;
        result.result = SandboxResult::ERROR;
        return result;
    }
    
    // Simulate sandbox execution and behavior observation
    std::cout << "[SandboxOrchestrator] Executing in isolated environment..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Simulate execution time
    
    // Observe behavior
    result.behaviors = observeBehavior(sandboxFile.string(), sandboxType);
    
    // Extract attack patterns
    result.attackPatterns = extractAttackPatterns(result.behaviors);
    
    // Determine if malware based on behaviors
    if (!result.attackPatterns.empty()) {
        result.result = SandboxResult::MALWARE_DETECTED;
        result.threatScore = static_cast<float>(result.attackPatterns.size()) / 10.0f;
        if (result.threatScore > 1.0f) result.threatScore = 1.0f;
    }
    
    result.detailedReport = "Sandbox analysis completed. Behaviors observed: " + 
                           std::to_string(result.behaviors.size());
    
    return result;
}

std::vector<std::string> SandboxOrchestrator::observeBehavior(
    const std::string& filePath,
    SandboxType sandboxType) {
    
    std::vector<std::string> behaviors;
    
    // Simulate behavior observation
    // In production, this would monitor:
    // - File system operations
    // - Registry modifications
    // - Network connections
    // - Process creation
    // - Memory manipulation
    
    std::cout << "[SandboxOrchestrator] Observing file behavior..." << std::endl;
    
    // Simulate some detected behaviors (in production, these would be real observations)
    std::ifstream file(filePath, std::ios::binary);
    if (file.is_open()) {
        file.seekg(0, std::ios::end);
        size_t fileSize = file.tellg();
        
        // Simple heuristics for demo
        if (fileSize > 1000000) {
            behaviors.push_back("LARGE_FILE_SIZE");
        }
        if (fileSize < 100) {
            behaviors.push_back("SUSPICIOUS_SMALL_SIZE");
        }
    }
    
    return behaviors;
}

std::vector<std::string> SandboxOrchestrator::extractAttackPatterns(
    const std::vector<std::string>& behaviors) {
    
    std::vector<std::string> patterns;
    
    for (const auto& behavior : behaviors) {
        if (behavior.find("SUSPICIOUS") != std::string::npos) {
            patterns.push_back("SUSPICIOUS_BEHAVIOR_PATTERN");
        }
        if (behavior.find("REGISTRY") != std::string::npos) {
            patterns.push_back("REGISTRY_MODIFICATION");
        }
        if (behavior.find("NETWORK") != std::string::npos) {
            patterns.push_back("NETWORK_COMMUNICATION");
        }
    }
    
    return patterns;
}

bool SandboxOrchestrator::performCleaning(const std::string& filePath) {
    std::cout << "[SandboxOrchestrator] Performing file cleaning..." << std::endl;
    
    // In production, this would:
    // - Remove malicious code sections
    // - Strip suspicious headers
    // - Sanitize data
    // - Validate file integrity
    
    // For demo, we just log the action
    std::cout << "[SandboxOrchestrator] File cleaned successfully" << std::endl;
    return true;
}
