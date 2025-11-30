/**
 * @file InferenceEngine.hpp
 * @brief ML inference engine for DRLHSS Antivirus
 * 
 * Performs ML inference using ONNX Runtime
 * Supports both static (PE features) and dynamic (behavior) models
 */

#ifndef DRLHSS_INFERENCE_ENGINE_HPP
#define DRLHSS_INFERENCE_ENGINE_HPP

#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

namespace drlhss {
namespace detection {
namespace av {

/**
 * @brief Prediction result from ML model
 */
struct PredictionResult {
    float benign_probability;
    float malicious_probability;
    bool is_malicious;
    float confidence;
    std::string verdict;
    
    PredictionResult() : benign_probability(0.0f),
                        malicious_probability(0.0f),
                        is_malicious(false),
                        confidence(0.0f),
                        verdict("UNKNOWN") {}
};

/**
 * @brief InferenceEngine - ML model inference using ONNX Runtime
 */
class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();
    
    /**
     * @brief Load static analysis model
     * @param model_path Path to ONNX model file
     * @return true if loaded successfully
     */
    bool loadStaticModel(const std::string& model_path);
    
    /**
     * @brief Load dynamic behavior model
     * @param model_path Path to ONNX model file
     * @return true if loaded successfully
     */
    bool loadDynamicModel(const std::string& model_path);
    
    /**
     * @brief Predict using static model
     * @param features Feature vector (size 2381)
     * @return Prediction result
     */
    PredictionResult predictStatic(const std::vector<float>& features);
    
    /**
     * @brief Predict using dynamic model
     * @param features Feature vector (size 500)
     * @return Prediction result
     */
    PredictionResult predictDynamic(const std::vector<float>& features);
    
    /**
     * @brief Combined prediction (60% static + 40% dynamic)
     * @param static_features Static feature vector
     * @param dynamic_features Dynamic feature vector
     * @return Combined prediction result
     */
    PredictionResult predictHybrid(const std::vector<float>& static_features,
                                   const std::vector<float>& dynamic_features);
    
    /**
     * @brief Check if models are loaded
     */
    bool isStaticModelLoaded() const { return static_model_loaded_; }
    bool isDynamicModelLoaded() const { return dynamic_model_loaded_; }
    
    /**
     * @brief Get last error
     */
    std::string getLastError() const { return last_error_; }
    
private:
    // ONNX Runtime components
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> static_session_;
    std::unique_ptr<Ort::Session> dynamic_session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    
    bool static_model_loaded_;
    bool dynamic_model_loaded_;
    std::string last_error_;
    
    // Model metadata
    int static_n_features_;
    int dynamic_n_features_;
    
    // Input/output names
    std::vector<const char*> static_input_names_;
    std::vector<const char*> static_output_names_;
    std::vector<const char*> dynamic_input_names_;
    std::vector<const char*> dynamic_output_names_;
    
    // Helper methods
    PredictionResult performInference(Ort::Session* session,
                                     const std::vector<float>& features,
                                     const std::vector<const char*>& input_names,
                                     const std::vector<const char*>& output_names,
                                     int expected_features);
    void interpretPrediction(PredictionResult& result);
    bool validateFeatures(const std::vector<float>& features, 
                         int expected_size);
};

} // namespace av
} // namespace detection
} // namespace drlhss

#endif // DRLHSS_INFERENCE_ENGINE_HPP
