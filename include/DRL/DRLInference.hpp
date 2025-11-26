#pragma once

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <atomic>

namespace drl {

/**
 * @brief Production-grade ONNX model inference for DRL
 * 
 * Thread-safe ONNX Runtime wrapper for real-time inference
 * with model hot-reloading and performance monitoring.
 */
class DRLInference {
public:
    /**
     * @brief Constructor
     * @param model_path Path to ONNX model file
     * @param num_threads Number of inference threads
     */
    explicit DRLInference(const std::string& model_path, int num_threads = 1);
    
    /**
     * @brief Destructor
     */
    ~DRLInference();
    
    /**
     * @brief Select action using policy network
     * @param state Input state vector
     * @return Selected action index
     */
    int selectAction(const std::vector<float>& state);
    
    /**
     * @brief Get Q-values for all actions
     * @param state Input state vector
     * @return Q-values for each action
     */
    std::vector<float> getQValues(const std::vector<float>& state);
    
    /**
     * @brief Hot-reload model from new file
     * @param new_model_path Path to new ONNX model
     * @return True if reload successful
     */
    bool reloadModel(const std::string& new_model_path);
    
    /**
     * @brief Get inference statistics
     */
    struct InferenceStats {
        uint64_t total_inferences = 0;
        double avg_latency_ms = 0.0;
        double max_latency_ms = 0.0;
        uint64_t errors = 0;
    };
    
    InferenceStats getStats() const;
    
    /**
     * @brief Reset statistics
     */
    void resetStats();
    
    /**
     * @brief Check if model is loaded and ready
     */
    bool isReady() const { return model_loaded_.load(); }
    
    /**
     * @brief Get input dimension
     */
    int getInputDim() const { return input_dim_; }
    
    /**
     * @brief Get output dimension
     */
    int getOutputDim() const { return output_dim_; }

private:
    // ONNX Runtime components
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
    
    // Model metadata
    std::string model_path_;
    int input_dim_;
    int output_dim_;
    std::atomic<bool> model_loaded_{false};
    
    // Thread safety
    mutable std::mutex inference_mutex_;
    mutable std::mutex stats_mutex_;
    
    // Performance statistics
    mutable InferenceStats stats_;
    
    // Helper methods
    bool loadModel(const std::string& path);
    void updateStats(double latency_ms, bool error = false);
    std::vector<float> runInference(const std::vector<float>& input);
};

} // namespace drl
