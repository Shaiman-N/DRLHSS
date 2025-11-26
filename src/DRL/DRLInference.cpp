#include "DRL/DRLInference.hpp"
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace drl {

DRLInference::DRLInference(const std::string& model_path, int num_threads)
    : model_path_(model_path), input_dim_(0), output_dim_(0) {
    
    // Initialize ONNX Runtime environment
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "DRLInference");
    
    // Configure session options
    session_options_ = std::make_unique<Ort::SessionOptions>();
    session_options_->SetIntraOpNumThreads(num_threads);
    session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    // Initialize memory info
    memory_info_ = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    
    // Load initial model
    if (!loadModel(model_path)) {
        throw std::runtime_error("Failed to load ONNX model: " + model_path);
    }
    
    std::cout << "[DRLInference] Initialized with model: " << model_path << std::endl;
    std::cout << "[DRLInference] Input dim: " << input_dim_ << ", Output dim: " << output_dim_ << std::endl;
}

DRLInference::~DRLInference() {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    session_.reset();
    session_options_.reset();
    memory_info_.reset();
    env_.reset();
}

int DRLInference::selectAction(const std::vector<float>& state) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        std::vector<float> q_values = getQValues(state);
        
        // Find action with maximum Q-value
        auto max_it = std::max_element(q_values.begin(), q_values.end());
        int action = static_cast<int>(std::distance(q_values.begin(), max_it));
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        updateStats(duration.count() / 1000.0);
        
        return action;
        
    } catch (const std::exception& e) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        updateStats(duration.count() / 1000.0, true);
        
        std::cerr << "[DRLInference] Error in selectAction: " << e.what() << std::endl;
        return 0; // Default action
    }
}

std::vector<float> DRLInference::getQValues(const std::vector<float>& state) {
    if (!model_loaded_.load()) {
        throw std::runtime_error("Model not loaded");
    }
    
    if (static_cast<int>(state.size()) != input_dim_) {
        throw std::invalid_argument("State dimension mismatch. Expected: " + 
                                   std::to_string(input_dim_) + ", Got: " + 
                                   std::to_string(state.size()));
    }
    
    return runInference(state);
}

bool DRLInference::reloadModel(const std::string& new_model_path) {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    
    try {
        // Create new session
        auto new_session = std::make_unique<Ort::Session>(*env_, new_model_path.c_str(), *session_options_);
        
        // Validate input/output dimensions
        auto input_info = new_session->GetInputTypeInfo(0);
        auto input_tensor_info = input_info.GetTensorTypeAndShapeInfo();
        auto input_shape = input_tensor_info.GetShape();
        
        auto output_info = new_session->GetOutputTypeInfo(0);
        auto output_tensor_info = output_info.GetTensorTypeAndShapeInfo();
        auto output_shape = output_tensor_info.GetShape();
        
        int new_input_dim = static_cast<int>(input_shape[1]);
        int new_output_dim = static_cast<int>(output_shape[1]);
        
        // Check compatibility
        if (input_dim_ != 0 && new_input_dim != input_dim_) {
            std::cerr << "[DRLInference] Input dimension mismatch in new model" << std::endl;
            return false;
        }
        
        if (output_dim_ != 0 && new_output_dim != output_dim_) {
            std::cerr << "[DRLInference] Output dimension mismatch in new model" << std::endl;
            return false;
        }
        
        // Replace session
        session_ = std::move(new_session);
        model_path_ = new_model_path;
        input_dim_ = new_input_dim;
        output_dim_ = new_output_dim;
        model_loaded_.store(true);
        
        std::cout << "[DRLInference] Model reloaded: " << new_model_path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[DRLInference] Failed to reload model: " << e.what() << std::endl;
        return false;
    }
}

DRLInference::InferenceStats DRLInference::getStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void DRLInference::resetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = InferenceStats{};
}

bool DRLInference::loadModel(const std::string& path) {
    try {
        session_ = std::make_unique<Ort::Session>(*env_, path.c_str(), *session_options_);
        
        // Get input/output dimensions
        auto input_info = session_->GetInputTypeInfo(0);
        auto input_tensor_info = input_info.GetTensorTypeAndShapeInfo();
        auto input_shape = input_tensor_info.GetShape();
        
        auto output_info = session_->GetOutputTypeInfo(0);
        auto output_tensor_info = output_info.GetTensorTypeAndShapeInfo();
        auto output_shape = output_tensor_info.GetShape();
        
        input_dim_ = static_cast<int>(input_shape[1]);
        output_dim_ = static_cast<int>(output_shape[1]);
        
        model_loaded_.store(true);
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[DRLInference] Failed to load model: " << e.what() << std::endl;
        model_loaded_.store(false);
        return false;
    }
}

void DRLInference::updateStats(double latency_ms, bool error) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    stats_.total_inferences++;
    
    if (error) {
        stats_.errors++;
    } else {
        // Update latency statistics
        double total_latency = stats_.avg_latency_ms * (stats_.total_inferences - 1);
        stats_.avg_latency_ms = (total_latency + latency_ms) / stats_.total_inferences;
        stats_.max_latency_ms = std::max(stats_.max_latency_ms, latency_ms);
    }
}

std::vector<float> DRLInference::runInference(const std::vector<float>& input) {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    
    // Create input tensor
    std::array<int64_t, 2> input_shape{1, input_dim_};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        *memory_info_,
        const_cast<float*>(input.data()),
        input.size(),
        input_shape.data(),
        input_shape.size()
    );
    
    // Run inference
    const char* input_names[] = {"state"};
    const char* output_names[] = {"q_values"};
    
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );
    
    // Extract output
    float* output_data = output_tensors.front().GetTensorMutableData<float>();
    std::vector<float> q_values(output_data, output_data + output_dim_);
    
    return q_values;
}

} // namespace drl
