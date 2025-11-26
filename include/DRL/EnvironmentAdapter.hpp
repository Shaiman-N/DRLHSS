#pragma once

#include "DRL/TelemetryData.hpp"
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>

namespace drl {

/**
 * @brief Environment adapter for converting telemetry to state vectors
 * 
 * Normalizes raw telemetry data into fixed-dimension state vectors
 * suitable for neural network input with consistent feature ordering.
 */
class EnvironmentAdapter {
public:
    /**
     * @brief Constructor
     * @param feature_dim Dimension of output state vector
     */
    explicit EnvironmentAdapter(int feature_dim);
    
    /**
     * @brief Process telemetry data into normalized state vector
     * @param telemetry Raw telemetry data
     * @return Normalized state vector
     */
    std::vector<float> processTelemetry(const TelemetryData& telemetry);
    
    /**
     * @brief Handle missing or malformed fields gracefully
     * @param telemetry Potentially incomplete telemetry
     * @return Complete telemetry with defaults filled
     */
    TelemetryData handleMissingFields(const TelemetryData& telemetry);
    
    /**
     * @brief Get feature dimension
     */
    int getFeatureDim() const { return feature_dim_; }
    
    /**
     * @brief Update normalization parameters from training data
     * @param telemetry_samples Vector of telemetry samples for statistics
     */
    void updateNormalizationParams(const std::vector<TelemetryData>& telemetry_samples);

private:
    int feature_dim_;
    std::vector<std::string> feature_names_;
    std::unordered_map<std::string, float> default_values_;
    
    struct NormalizationParams {
        float min_val = 0.0f;
        float max_val = 1.0f;
        float mean = 0.0f;
        float std_dev = 1.0f;
    };
    
    std::unordered_map<std::string, NormalizationParams> normalization_params_;
    
    void initializeDefaults();
    void initializeNormalization();
    float normalizeFeature(const std::string& feature_name, float value);
    std::vector<float> extractDerivedFeatures(const TelemetryData& telemetry);
};

} // namespace drl
