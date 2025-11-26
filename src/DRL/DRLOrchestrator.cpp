#include "DRL/DRLOrchestrator.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>

namespace drl {

DRLOrchestrator::DRLOrchestrator(const std::string& model_path,
                                 const std::string& db_path,
                                 int feature_dim)
    : model_path_(model_path), db_path_(db_path), feature_dim_(feature_dim) {
}

DRLOrchestrator::~DRLOrchestrator() {
    stopPatternLearning();
}

bool DRLOrchestrator::initialize() {
    try {
        // Initialize inference engine
        inference_ = std::make_unique<DRLInference>(model_path_);
        if (!inference_->isReady()) {
            std::cerr << "[DRLOrchestrator] Failed to initialize inference engine" << std::endl;
            return false;
        }
        
        // Initialize environment adapter
        adapter_ = std::make_unique<EnvironmentAdapter>(feature_dim_);
        
        // Initialize replay buffer
        replay_buffer_ = std::make_unique<ReplayBuffer>(100000);
        
        // Initialize database
        db_manager_ = std::make_unique<db::DatabaseManager>(db_path_);
        if (!db_manager_->initialize()) {
            std::cerr << "[DRLOrchestrator] Failed to initialize database" << std::endl;
            return false;
        }
        
        std::cout << "[DRLOrchestrator] Initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[DRLOrchestrator] Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

int DRLOrchestrator::processAndDecide(const TelemetryData& telemetry) {
    total_detections_++;
    
    // Convert telemetry to state vector
    std::vector<float> state = adapter_->processTelemetry(telemetry);
    
    // Get action from policy network
    int action = inference_->selectAction(state);
    
    // Store telemetry in database
    db_manager_->storeTelemetry(telemetry);
    
    return action;
}

DRLOrchestrator::DetectionResponse DRLOrchestrator::processWithDetails(const TelemetryData& telemetry) {
    total_detections_++;
    
    DetectionResponse response;
    
    // Convert telemetry to state vector
    std::vector<float> state = adapter_->processTelemetry(telemetry);
    
    // Get Q-values from policy network
    response.q_values = inference_->getQValues(state);
    
    // Select action (highest Q-value)
    auto max_it = std::max_element(response.q_values.begin(), response.q_values.end());
    response.action = static_cast<int>(std::distance(response.q_values.begin(), max_it));
    
    // Compute confidence
    response.confidence = computeConfidence(response.q_values);
    
    // Classify attack type
    response.attack_type = classifyAttackType(telemetry, response.action);
    
    // Determine if malicious
    response.is_malicious = (response.action == 1 || response.action == 2);  // Block or Quarantine
    
    if (response.is_malicious) {
        malicious_detected_++;
    }
    
    // Store telemetry in database
    db_manager_->storeTelemetry(telemetry);
    
    return response;
}


void DRLOrchestrator::storeExperience(const TelemetryData& telemetry, int action, float reward,
                                      const TelemetryData& next_telemetry, bool done) {
    std::vector<float> state = adapter_->processTelemetry(telemetry);
    std::vector<float> next_state = adapter_->processTelemetry(next_telemetry);
    
    Experience exp(state, action, reward, next_state, done);
    replay_buffer_->add(exp);
    
    // Store in database with episode ID
    std::string episode_id = telemetry.sandbox_id + "_" + 
                            std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    db_manager_->storeExperience(exp, episode_id);
}

void DRLOrchestrator::learnAttackPattern(const TelemetryData& telemetry, int action, float reward,
                                         const std::string& attack_type, float confidence) {
    AttackPattern pattern;
    pattern.telemetry_features = adapter_->processTelemetry(telemetry);
    pattern.action_taken = action;
    pattern.reward = reward;
    pattern.attack_type = attack_type;
    pattern.confidence_score = confidence;
    pattern.timestamp = std::chrono::system_clock::now();
    pattern.sandbox_id = telemetry.sandbox_id;
    pattern.artifact_hash = telemetry.artifact_hash;
    
    if (pattern.isValid()) {
        db_manager_->storeAttackPattern(pattern);
    }
}

bool DRLOrchestrator::reloadModel(const std::string& new_model_path) {
    if (inference_->reloadModel(new_model_path)) {
        model_path_ = new_model_path;
        std::cout << "[DRLOrchestrator] Model reloaded successfully" << std::endl;
        return true;
    }
    return false;
}

DRLOrchestrator::SystemStats DRLOrchestrator::getStats() const {
    SystemStats stats;
    stats.total_detections = total_detections_.load();
    stats.malicious_detected = malicious_detected_.load();
    stats.false_positives = false_positives_.load();
    stats.replay_buffer_size = replay_buffer_->size();
    stats.db_stats = db_manager_->getStats();
    
    auto inference_stats = inference_->getStats();
    stats.avg_inference_time_ms = inference_stats.avg_latency_ms;
    
    return stats;
}

bool DRLOrchestrator::exportExperiences(const std::string& output_path, int limit) {
    try {
        std::ofstream file(output_path);
        if (!file.is_open()) {
            return false;
        }
        
        // Export experiences from replay buffer
        size_t count = std::min(static_cast<size_t>(limit), replay_buffer_->size());
        auto experiences = replay_buffer_->sample(count);
        
        nlohmann::json j;
        j["count"] = count;
        j["experiences"] = nlohmann::json::array();
        
        for (const auto& exp : experiences) {
            nlohmann::json exp_json;
            exp_json["state"] = exp.state;
            exp_json["action"] = exp.action;
            exp_json["reward"] = exp.reward;
            exp_json["next_state"] = exp.next_state;
            exp_json["done"] = exp.done;
            j["experiences"].push_back(exp_json);
        }
        
        file << j.dump(4);
        file.close();
        
        std::cout << "[DRLOrchestrator] Exported " << count << " experiences to " << output_path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[DRLOrchestrator] Failed to export experiences: " << e.what() << std::endl;
        return false;
    }
}

void DRLOrchestrator::startPatternLearning() {
    if (learning_active_.load()) {
        return;
    }
    
    learning_active_.store(true);
    learning_thread_ = std::make_unique<std::thread>(&DRLOrchestrator::backgroundPatternLearning, this);
    
    std::cout << "[DRLOrchestrator] Started background pattern learning" << std::endl;
}

void DRLOrchestrator::stopPatternLearning() {
    if (!learning_active_.load()) {
        return;
    }
    
    learning_active_.store(false);
    if (learning_thread_ && learning_thread_->joinable()) {
        learning_thread_->join();
    }
    
    std::cout << "[DRLOrchestrator] Stopped background pattern learning" << std::endl;
}

std::string DRLOrchestrator::classifyAttackType(const TelemetryData& telemetry, int action) {
    // Simple heuristic-based classification
    if (telemetry.code_injection_detected) {
        return "code_injection";
    }
    if (telemetry.privilege_escalation_attempt) {
        return "privilege_escalation";
    }
    if (telemetry.registry_modification && telemetry.file_write_count > 10) {
        return "ransomware";
    }
    if (telemetry.network_connections > 20 && telemetry.bytes_sent > 1000000) {
        return "data_exfiltration";
    }
    if (telemetry.child_processes > 5) {
        return "process_injection";
    }
    if (telemetry.file_delete_count > 10) {
        return "destructive_malware";
    }
    
    return action == 0 ? "benign" : "suspicious";
}

float DRLOrchestrator::computeConfidence(const std::vector<float>& q_values) {
    if (q_values.empty()) return 0.0f;
    
    // Softmax to get probabilities
    std::vector<float> exp_values;
    float sum = 0.0f;
    
    for (float q : q_values) {
        float exp_q = std::exp(q);
        exp_values.push_back(exp_q);
        sum += exp_q;
    }
    
    // Max probability as confidence
    float max_prob = 0.0f;
    for (float exp_q : exp_values) {
        max_prob = std::max(max_prob, exp_q / sum);
    }
    
    return max_prob;
}

void DRLOrchestrator::backgroundPatternLearning() {
    while (learning_active_.load()) {
        // Periodically analyze replay buffer for patterns
        if (replay_buffer_->size() > 1000) {
            // Sample experiences and look for patterns
            auto experiences = replay_buffer_->sample(100);
            
            // Analyze high-reward experiences
            for (const auto& exp : experiences) {
                if (exp.reward > 0.8f) {
                    // This is a good detection - could learn from it
                    // In production, this would trigger more sophisticated pattern analysis
                }
            }
        }
        
        // Sleep for a while
        std::this_thread::sleep_for(std::chrono::seconds(60));
    }
}

} // namespace drl
