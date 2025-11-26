#include "DRL/ModelMetadata.hpp"
#include <fstream>
#include <stdexcept>

namespace drl {

nlohmann::json ModelMetadata::toJson() const {
    nlohmann::json j;
    
    // Version information
    j["model_version"] = model_version;
    j["training_date"] = std::chrono::duration_cast<std::chrono::milliseconds>(
        training_date.time_since_epoch()).count();
    
    // Training statistics
    j["training_episodes"] = training_episodes;
    j["final_average_reward"] = final_average_reward;
    j["final_loss"] = final_loss;
    
    // Hyperparameters
    j["learning_rate"] = learning_rate;
    j["gamma"] = gamma;
    j["epsilon_start"] = epsilon_start;
    j["epsilon_end"] = epsilon_end;
    j["batch_size"] = batch_size;
    j["target_update_frequency"] = target_update_frequency;
    
    // Performance metrics
    j["detection_accuracy"] = detection_accuracy;
    j["false_positive_rate"] = false_positive_rate;
    j["false_negative_rate"] = false_negative_rate;
    
    // Model architecture
    j["input_dim"] = input_dim;
    j["output_dim"] = output_dim;
    j["hidden_layers"] = hidden_layers;
    
    return j;
}

ModelMetadata ModelMetadata::fromJson(const nlohmann::json& j) {
    ModelMetadata metadata;
    
    try {
        // Version information
        metadata.model_version = j.value("model_version", "");
        if (j.contains("training_date")) {
            auto ms = j["training_date"].get<int64_t>();
            metadata.training_date = std::chrono::system_clock::time_point(
                std::chrono::milliseconds(ms));
        }
        
        // Training statistics
        metadata.training_episodes = j.value("training_episodes", 0);
        metadata.final_average_reward = j.value("final_average_reward", 0.0f);
        metadata.final_loss = j.value("final_loss", 0.0f);
        
        // Hyperparameters
        metadata.learning_rate = j.value("learning_rate", 0.0001f);
        metadata.gamma = j.value("gamma", 0.99f);
        metadata.epsilon_start = j.value("epsilon_start", 1.0f);
        metadata.epsilon_end = j.value("epsilon_end", 0.1f);
        metadata.batch_size = j.value("batch_size", 64);
        metadata.target_update_frequency = j.value("target_update_frequency", 1000);
        
        // Performance metrics
        metadata.detection_accuracy = j.value("detection_accuracy", 0.0f);
        metadata.false_positive_rate = j.value("false_positive_rate", 0.0f);
        metadata.false_negative_rate = j.value("false_negative_rate", 0.0f);
        
        // Model architecture
        metadata.input_dim = j.value("input_dim", 0);
        metadata.output_dim = j.value("output_dim", 0);
        metadata.hidden_layers = j.value("hidden_layers", std::vector<int>{});
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to parse ModelMetadata from JSON: " + 
                                 std::string(e.what()));
    }
    
    return metadata;
}

bool ModelMetadata::saveToFile(const std::string& filepath) const {
    try {
        std::ofstream file(filepath);
        if (!file.is_open()) {
            return false;
        }
        
        nlohmann::json j = toJson();
        file << j.dump(4);  // Pretty print with 4-space indentation
        file.close();
        return true;
        
    } catch (const std::exception&) {
        return false;
    }
}

ModelMetadata ModelMetadata::loadFromFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open metadata file: " + filepath);
    }
    
    nlohmann::json j;
    file >> j;
    file.close();
    
    return fromJson(j);
}

} // namespace drl
