#include "drl_framework.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace nidps {

DRLFramework::DRLFramework(const std::string& model_path)
    : model_path_(model_path), initialized_(false) {}

DRLFramework::~DRLFramework() {}

bool DRLFramework::initialize() {
    try {
        ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "NIDPS_DRL");
        session_options_ = std::make_unique<Ort::SessionOptions>();
        
        session_options_->SetIntraOpNumThreads(4);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        ort_session_ = std::make_unique<Ort::Session>(*ort_env_, model_path_.c_str(), *session_options_);
        
        // Get input/output information
        Ort::AllocatorWithDefaultOptions allocator;
        
        size_t num_input_nodes = ort_session_->GetInputCount();
        size_t num_output_nodes = ort_session_->GetOutputCount();
        
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = ort_session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(strdup(input_name.get()));
            
            Ort::TypeInfo type_info = ort_session_->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            input_shape_ = tensor_info.GetShape();
        }
        
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = ort_session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(strdup(output_name.get()));
            
            Ort::TypeInfo type_info = ort_session_->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            output_shape_ = tensor_info.GetShape();
        }
        
        initialized_ = true;
        std::cout << "DRL Framework initialized with model: " << model_path_ << std::endl;
        std::cout << "Input shape: [";
        for (auto dim : input_shape_) std::cout << dim << " ";
        std::cout << "]" << std::endl;
        
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return false;
    }
}

bool DRLFramework::detectMalware(const PacketPtr& packet, float& confidence) {
    if (!initialized_) {
        std::cerr << "DRL Framework not initialized" << std::endl;
        return false;
    }
    
    std::vector<float> features = extractFeatures(packet);
    preprocessFeatures(features);
    
    std::vector<float> output = runInference(features);
    
    if (!output.empty()) {
        confidence = output[0];
        return confidence > 0.5f;
    }
    
    return false;
}

AttackPattern DRLFramework::learnFromBehavior(const PacketPtr& packet, 
                                              const SandboxBehavior& behavior) {
    AttackPattern pattern;
    pattern.feature_vector = extractFeatures(packet);
    pattern.behavior = behavior;
    pattern.attack_type = classifyAttack(behavior);
    pattern.confidence = static_cast<float>(behavior.threat_score) / 100.0f;
    pattern.learned_at = std::chrono::system_clock::now();
    
    std::cout << "Learned attack pattern: " << pattern.attack_type 
              << " (confidence: " << pattern.confidence << ")" << std::endl;
    
    return pattern;
}

void DRLFramework::updateModel(const std::vector<AttackPattern>& patterns) {
    // In production, this would trigger model retraining
    // For now, we log the patterns for offline training
    std::cout << "Model update requested with " << patterns.size() 
              << " new patterns" << std::endl;
}

std::vector<float> DRLFramework::extractFeatures(const PacketPtr& packet) {
    if (!packet->features.empty()) {
        return packet->features;
    }
    
    std::vector<float> features;
    features.reserve(20);
    
    features.push_back(static_cast<float>(packet->raw_data.size()));
    features.push_back(static_cast<float>(packet->source_port));
    features.push_back(static_cast<float>(packet->dest_port));
    features.push_back(static_cast<float>(packet->protocol));
    
    // Statistical features from raw data
    if (!packet->raw_data.empty()) {
        float sum = 0;
        for (uint8_t byte : packet->raw_data) {
            sum += byte;
        }
        float mean = sum / packet->raw_data.size();
        features.push_back(mean);
        
        float variance = 0;
        for (uint8_t byte : packet->raw_data) {
            variance += (byte - mean) * (byte - mean);
        }
        variance /= packet->raw_data.size();
        features.push_back(std::sqrt(variance));
    } else {
        features.push_back(0.0f);
        features.push_back(0.0f);
    }
    
    // Pad to expected size
    while (features.size() < 20) {
        features.push_back(0.0f);
    }
    
    return features;
}

std::string DRLFramework::classifyAttack(const SandboxBehavior& behavior) {
    if (behavior.memory_injection) {
        return "MEMORY_INJECTION";
    }
    if (behavior.registry_modified && behavior.process_created) {
        return "TROJAN";
    }
    if (behavior.network_activity && behavior.suspicious_api_calls) {
        return "BOTNET";
    }
    if (behavior.file_system_modified) {
        return "RANSOMWARE";
    }
    if (behavior.process_created) {
        return "BACKDOOR";
    }
    
    return "UNKNOWN_MALWARE";
}

void DRLFramework::preprocessFeatures(std::vector<float>& features) {
    normalizeFeatures(features);
    
    // Ensure correct input size
    int64_t expected_size = 1;
    for (auto dim : input_shape_) {
        if (dim > 0) expected_size *= dim;
    }
    
    if (features.size() < static_cast<size_t>(expected_size)) {
        features.resize(expected_size, 0.0f);
    } else if (features.size() > static_cast<size_t>(expected_size)) {
        features.resize(expected_size);
    }
}

std::vector<float> DRLFramework::runInference(const std::vector<float>& input_features) {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    
    try {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        
        std::vector<int64_t> input_shape = input_shape_;
        if (input_shape[0] == -1) {
            input_shape[0] = 1;
        }
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, 
            const_cast<float*>(input_features.data()), 
            input_features.size(),
            input_shape.data(), 
            input_shape.size()
        );
        
        auto output_tensors = ort_session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            &input_tensor,
            1,
            output_names_.data(),
            output_names_.size()
        );
        
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        size_t output_count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        
        return std::vector<float>(output_data, output_data + output_count);
        
    } catch (const Ort::Exception& e) {
        std::cerr << "Inference error: " << e.what() << std::endl;
        return {};
    }
}

void DRLFramework::normalizeFeatures(std::vector<float>& features) {
    if (features.empty()) return;
    
    // Min-max normalization
    float min_val = *std::min_element(features.begin(), features.end());
    float max_val = *std::max_element(features.begin(), features.end());
    
    if (max_val - min_val > 0.0001f) {
        for (float& f : features) {
            f = (f - min_val) / (max_val - min_val);
        }
    }
}

} // namespace nidps
