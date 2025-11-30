#pragma once

#include <string>
#include <vector>
#include <memory>

struct AttackPattern {
    std::string patternType;
    std::string description;
    std::vector<std::string> indicators;
    float severity;
    long long timestamp;
};

struct BehaviorData {
    std::string filePath;
    std::vector<std::string> behaviors;
    std::vector<AttackPattern> patterns;
    float threatLevel;
    std::string sandboxType;
    long long timestamp;
};

class DRLFramework {
public:
    DRLFramework(const std::string& databasePath);
    ~DRLFramework();
    
    // Learn from attack patterns and behaviors
    void learnFromBehavior(const BehaviorData& behaviorData);
    
    // Store learned information to database
    bool storeToDatabase(const BehaviorData& behaviorData);
    
    // Update DRL model with new patterns
    void updateModel(const std::vector<AttackPattern>& patterns);
    
    // Get threat prediction based on learned patterns
    float predictThreatLevel(const std::vector<std::string>& behaviors);
    
    // Retrieve similar attack patterns from database
    std::vector<AttackPattern> getSimilarPatterns(const std::vector<std::string>& behaviors);
    
private:
    std::string databasePath_;
    
    // Analyze behavior patterns
    std::vector<AttackPattern> analyzeBehaviors(const std::vector<std::string>& behaviors);
    
    // Calculate reward for reinforcement learning
    float calculateReward(const BehaviorData& behaviorData);
    
    // Update Q-values or policy network
    void updateLearningModel(float reward, const BehaviorData& behaviorData);
};
