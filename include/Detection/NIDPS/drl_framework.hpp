#ifndef DRL_FRAMEWORK_HPP
#define DRL_FRAMEWORK_HPP

#include "packet_data.hpp"
#include "sandbox.hpp"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>
#include <mutex>

namespace nidps {

struct AttackPattern {
    std::vector<float> feature_vector;
    std::string attack_type;
    float confidence;
    SandboxBehavior behavior;
    std::chrono::system_clock::time_point learned_at;
    
    AttackPattern() : confidence(0.0f) {
        learned_at = std::chrono::system_clock::now();
    }
};

class DRLFramework {
public:
    DRLFramework(const std::string& model_path);
    ~DRLFramework();
    
    bool initialize();
    
    bool detectMalware(const PacketPtr& packet, float& confidence);
    AttackPattern learnFromBehavior(const PacketPtr& packet, const SandboxBehavior& behavior);
    void updateModel(const std::vector<AttackPattern>& patterns);
    
    std::vector<float> extractFeatures(const PacketPtr& packet);
    std::string classifyAttack(const SandboxBehavior& behavior);
    
private:
    void preprocessFeatures(std::vector<float>& features);
    std::vector<float> runInference(const std::vector<float>& input_features);
    void normalizeFeatures(std::vector<float>& features);
    
    std::string model_path_;
    std::unique_ptr<Ort::Env> ort_env_;
    std::unique_ptr<Ort::Session> ort_session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
    
    std::mutex inference_mutex_;
    bool initialized_;
};

} // namespace nidps

#endif // DRL_FRAMEWORK_HPP
