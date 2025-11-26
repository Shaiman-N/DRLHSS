#pragma once
#include <string>
#include <vector>

class FeatureExtractor
{
    public:
        FeatureExtractor();

        // Extract numerical features from processed packet (stub with dummy features)
        std::vector<float> extractFeatures(const std::string& processed_packet);
};