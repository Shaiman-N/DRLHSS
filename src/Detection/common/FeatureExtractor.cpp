#include "FeatureExtractor.hpp"

FeatureExtractor::FeatureExtractor() {}

std::vector<float> FeatureExtractor::extractFeatures(const std::string& processed_packet) 
{
    // Dummy feature : lenght of string, count of 'a', count of 'e'
    float len = static_cast<float>(processed_packet.size());
    float count_a = static_cast<float>(std::count(processed_packet.begin(), processed_packet.end(), 'a'));
    float count_e = static_cast<float>(std::count(processed_packet.begin(), processed_packet.end(), 'a'));
    return {len, count_a, count_e};float count_a = static_cast<float>(std::count(processed_packet.begin(), processed_packet.end(), 'a'));
}