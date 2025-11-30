#include "DRLFramework.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>

DRLFramework::DRLFramework(const std::string& databasePath)
    : databasePath_(databasePath) {
    std::cout << "[DRLFramework] Initialized with database: " << databasePath << std::endl;
}

DRLFramework::~DRLFramework() {
    std::cout << "[DRLFramework] Shutting down" << std::endl;
}

void DRLFramework::learnFromBehavior(const BehaviorData& behaviorData) {
    std::cout << "[DRLFramework] Learning from behavior data..." << std::endl;
    std::cout << "[DRLFramework] File: " << behaviorData.filePath << std::endl;
    std::cout << "[DRLFramework] Behaviors: " << behaviorData.behaviors.size() << std::endl;
    std::cout << "[DRLFramework] Patterns: " << behaviorData.patterns.size() << std::endl;
    std::cout << "[DRLFramework] Threat Level: " << behaviorData.threatLevel << std::endl;
    
    // Analyze behaviors to extract patterns
    std::vector<AttackPattern> analyzedPatterns = analyzeBehaviors(behaviorData.behaviors);
    
    // Calculate reward for reinforcement learning
    float reward = calculateReward(behaviorData);
    std::cout << "[DRLFramework] Calculated reward: " << reward << std::endl;
    
    // Update learning model
    updateLearningModel(reward, behaviorData);
    
    // Update model with new patterns
    updateModel(behaviorData.patterns);
    
    std::cout << "[DRLFramework] Learning completed" << std::endl;
}

bool DRLFramework::storeToDatabase(const BehaviorData& behaviorData) {
    std::cout << "[DRLFramework] Storing learned information to database..." << std::endl;
    
    try {
        // Create database directory if it doesn't exist
        std::filesystem::path dbPath(databasePath_);
        if (!std::filesystem::exists(dbPath.parent_path())) {
            std::filesystem::create_directories(dbPath.parent_path());
        }
        
        // Append to database file (in production, use proper database)
        std::ofstream dbFile(databasePath_, std::ios::app);
        if (!dbFile.is_open()) {
            std::cerr << "[DRLFramework] Failed to open database file" << std::endl;
            return false;
        }
        
        // Write behavior data
        dbFile << "=== BEHAVIOR RECORD ===" << std::endl;
        dbFile << "Timestamp: " << behaviorData.timestamp << std::endl;
        dbFile << "File: " << behaviorData.filePath << std::endl;
        dbFile << "Sandbox: " << behaviorData.sandboxType << std::endl;
        dbFile << "Threat Level: " << behaviorData.threatLevel << std::endl;
        
        dbFile << "Behaviors:" << std::endl;
        for (const auto& behavior : behaviorData.behaviors) {
            dbFile << "  - " << behavior << std::endl;
        }
        
        dbFile << "Attack Patterns:" << std::endl;
        for (const auto& pattern : behaviorData.patterns) {
            dbFile << "  - Type: " << pattern.patternType << std::endl;
            dbFile << "    Description: " << pattern.description << std::endl;
            dbFile << "    Severity: " << pattern.severity << std::endl;
        }
        
        dbFile << std::endl;
        dbFile.close();
        
        std::cout << "[DRLFramework] Data stored successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[DRLFramework] Error storing to database: " << e.what() << std::endl;
        return false;
    }
}

void DRLFramework::updateModel(const std::vector<AttackPattern>& patterns) {
    std::cout << "[DRLFramework] Updating DRL model with " << patterns.size() << " patterns" << std::endl;
    
    // In production, this would:
    // - Update neural network weights
    // - Adjust Q-values
    // - Update policy network
    // - Retrain on new patterns
    
    for (const auto& pattern : patterns) {
        std::cout << "[DRLFramework]   Pattern: " << pattern.patternType 
                  << " (Severity: " << pattern.severity << ")" << std::endl;
    }
    
    std::cout << "[DRLFramework] Model updated" << std::endl;
}

float DRLFramework::predictThreatLevel(const std::vector<std::string>& behaviors) {
    std::cout << "[DRLFramework] Predicting threat level..." << std::endl;
    
    // Simple heuristic for demo (in production, use trained model)
    float threatLevel = 0.0f;
    
    for (const auto& behavior : behaviors) {
        if (behavior.find("SUSPICIOUS") != std::string::npos) {
            threatLevel += 0.3f;
        }
        if (behavior.find("MALICIOUS") != std::string::npos) {
            threatLevel += 0.5f;
        }
        if (behavior.find("ATTACK") != std::string::npos) {
            threatLevel += 0.4f;
        }
    }
    
    if (threatLevel > 1.0f) threatLevel = 1.0f;
    
    std::cout << "[DRLFramework] Predicted threat level: " << threatLevel << std::endl;
    return threatLevel;
}

std::vector<AttackPattern> DRLFramework::getSimilarPatterns(
    const std::vector<std::string>& behaviors) {
    
    std::cout << "[DRLFramework] Searching for similar patterns in database..." << std::endl;
    
    std::vector<AttackPattern> similarPatterns;
    
    // In production, this would query the database for similar patterns
    // For demo, return empty vector
    
    std::cout << "[DRLFramework] Found " << similarPatterns.size() << " similar patterns" << std::endl;
    return similarPatterns;
}

std::vector<AttackPattern> DRLFramework::analyzeBehaviors(
    const std::vector<std::string>& behaviors) {
    
    std::vector<AttackPattern> patterns;
    
    for (const auto& behavior : behaviors) {
        AttackPattern pattern;
        pattern.patternType = behavior;
        pattern.description = "Analyzed from behavior: " + behavior;
        pattern.severity = 0.5f;
        pattern.timestamp = std::chrono::system_clock::now().time_since_epoch().count();
        patterns.push_back(pattern);
    }
    
    return patterns;
}

float DRLFramework::calculateReward(const BehaviorData& behaviorData) {
    // Reward calculation for reinforcement learning
    // Positive reward for detecting threats
    // Negative reward for false positives
    
    float reward = 0.0f;
    
    if (behaviorData.threatLevel > 0.7f) {
        reward = 1.0f;  // High reward for detecting serious threats
    } else if (behaviorData.threatLevel > 0.3f) {
        reward = 0.5f;  // Medium reward
    } else {
        reward = 0.1f;  // Small reward for clean files
    }
    
    // Bonus for finding new patterns
    reward += behaviorData.patterns.size() * 0.1f;
    
    return reward;
}

void DRLFramework::updateLearningModel(float reward, const BehaviorData& behaviorData) {
    std::cout << "[DRLFramework] Updating learning model with reward: " << reward << std::endl;
    
    // In production, this would:
    // - Update Q-values: Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
    // - Update policy network gradients
    // - Perform backpropagation
    // - Update experience replay buffer
    
    std::cout << "[DRLFramework] Learning model updated" << std::endl;
}
