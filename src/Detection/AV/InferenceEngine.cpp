/**
 * @file InferenceEngine.cpp
 * @brief Implementation of ML inference engine using ONNX Runtime for DRLHSS
 */

#include "Detection/AV/InferenceEngine.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace drlhss {
namespace detection {
namespace av {

InferenceEngine::InferenceEngine()
    : static_model_loaded_(false)
    , dynamic_model_loaded_(false)
    , static_n_features_(2381)
    , dynamic_n_features_(500)
{
    // Initialize ONNX Runtime environment
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "DRLHSS_AV");
    session_options_ = std::make_unique<Ort::SessionOptions>();
    session_options_->SetIntraOpNumThreads(1);
    session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
}

InferenceEngine::~InferenceEngine() {
}

bool InferenceEngine::loadStaticModel(const std::string& model_path) {
    try {
        std::cout << "[InferenceEngine] Loading static model: " << model_path << std::endl;
        
        #ifdef _WIN32
        std::wstring wmodel_path(model_path.begin(), model_path.end());
        static_session_ = std::make_unique<Ort::Session>(*env_, wmodel_path.c_str(), *session_options_);
        #else
        static_session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *session_options_);
        #endif
        
        // Get input/output names
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Input names
        size_t num_input_nodes = static_session_->GetInputCount();
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = static_session_->GetInputNameAllocated(i, allocator);
            static_input_names_.push_back(input_name.get());
        }
        
        // Output names
        size_t num_output_nodes = static_session_->GetOutputCount();
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = static_session_->GetOutputNameAllocated(i, allocator);
            static_output_names_.push_back(output_name.get());
        }
        
        static_model_loaded_ = true;
        std::cout << "[InferenceEngine] Static model loaded successfully" << std::endl;
        
        return true;
        
    } catch (const Ort::Exception& e) {
        last_error_ = "Failed to load static model: " + std::string(e.what());
        std::cerr << "[InferenceEngine] " << last_error_ << std::endl;
        return false;
    }
}

bool InferenceEngine::loadDynamicModel(const std::string& model_path) {
    try {
        std::cout << "[InferenceEngine] Loading dynamic model: " << model_path << std::endl;
        
        #ifdef _WIN32
        std::wstring wmodel_path(model_path.begin(), model_path.end());
        dynamic_session_ = std::make_unique<Ort::Session>(*env_, wmodel_path.c_str(), *session_options_);
        #else
        dynamic_session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *session_options_);
        #endif
        
        // Get input/output names
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Input names
        size_t num_input_nodes = dynamic_session_->GetInputCount();
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = dynamic_session_->GetInputNameAllocated(i, allocator);
            dynamic_input_names_.push_back(input_name.get());
        }
        
        // Output names
        size_t num_output_nodes = dynamic_session_->GetOutputCount();
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = dynamic_session_->GetOutputNameAllocated(i, allocator);
            dynamic_output_names_.push_back(output_name.get());
        }
        
        dynamic_model_loaded_ = true;
        std::cout << "[InferenceEngine] Dynamic model loaded successfully" << std::endl;
        
        return true;
        
    } catch (const Ort::Exception& e) {
        last_error_ = "Failed to load dynamic model: " + std::string(e.what());
        std::cerr << "[InferenceEngine] " << last_error_ << std::endl;
        return false;
    }
}

PredictionResult InferenceEngine::predictStatic(const std::vector<float>& features) {
    if (!static_model_loaded_) {
        PredictionResult result;
        result.verdict = "ERROR: Model not loaded";
        last_error_ = "Static model not loaded";
        return result;
    }
    
    return performInference(static_session_.get(), features, 
                           static_input_names_, static_output_names_,
                           static_n_features_);
}

PredictionResult InferenceEngine::predictDynamic(const std::vector<float>& features) {
    if (!dynamic_model_loaded_) {
        PredictionResult result;
        result.verdict = "ERROR: Model not loaded";
        last_error_ = "Dynamic model not loaded";
        return result;
    }
    
    return performInference(dynamic_session_.get(), features,
                           dynamic_input_names_, dynamic_output_names_,
                           dynamic_n_features_);
}

PredictionResult InferenceEngine::predictHybrid(const std::vector<float>& static_features,
                                                const std::vector<float>& dynamic_features) {
    auto static_result = predictStatic(static_features);
    auto dynamic_result = predictDynamic(dynamic_features);
    
    PredictionResult hybrid_result;
    
    // Combine: 60% static + 40% dynamic
    hybrid_result.benign_probability = (static_result.benign_probability * 0.6f) + 
                                       (dynamic_result.benign_probability * 0.4f);
    hybrid_result.malicious_probability = (static_result.malicious_probability * 0.6f) + 
                                          (dynamic_result.malicious_probability * 0.4f);
    
    interpretPrediction(hybrid_result);
    
    return hybrid_result;
}

PredictionResult InferenceEngine::performInference(Ort::Session* session,
                                                   const std::vector<float>& features,
                                                   const std::vector<const char*>& input_names,
                                                   const std::vector<const char*>& output_names,
                                                   int expected_features) {
    PredictionResult result;
    
    // Validate features
    if (!validateFeatures(features, expected_features)) {
        result.verdict = "ERROR: Invalid features";
        return result;
    }
    
    try {
        // Create input tensor
        std::vector<int64_t> input_shape = {1, expected_features};
        
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(features.data()),
            features.size(),
            input_shape.data(),
            input_shape.size()
        );
        
        // Run inference
        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            &input_tensor,
            1,
            output_names.data(),
            output_names.size()
        );
        
        // Get output
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        
        // Assuming binary classification output: [benign_score, malicious_score]
        // Apply softmax
        float exp_benign = std::exp(output_data[0]);
        float exp_malicious = std::exp(output_data[1]);
        float sum = exp_benign + exp_malicious;
        
        result.benign_probability = exp_benign / sum;
        result.malicious_probability = exp_malicious / sum;
        
        interpretPrediction(result);
        
    } catch (const Ort::Exception& e) {
        last_error_ = "Inference failed: " + std::string(e.what());
        result.verdict = "ERROR: Inference failed";
        std::cerr << "[InferenceEngine] " << last_error_ << std::endl;
    }
    
    return result;
}

void InferenceEngine::interpretPrediction(PredictionResult& result) {
    result.is_malicious = result.malicious_probability > 0.5f;
    result.confidence = result.malicious_probability * 100.0f;
    
    if (result.is_malicious) {
        result.verdict = "MALICIOUS";
    } else {
        result.verdict = "BENIGN";
    }
}

bool InferenceEngine::validateFeatures(const std::vector<float>& features, 
                                       int expected_size) {
    if (features.size() != static_cast<size_t>(expected_size)) {
        last_error_ = "Feature size mismatch: expected " + 
                     std::to_string(expected_size) + 
                     ", got " + std::to_string(features.size());
        return false;
    }
    
    // Check for NaN or Inf
    for (const auto& f : features) {
        if (std::isnan(f) || std::isinf(f)) {
            last_error_ = "Invalid feature value (NaN or Inf)";
            return false;
        }
    }
    
    return true;
}

} // namespace av
} // namespace detection
} // namespace drlhss
