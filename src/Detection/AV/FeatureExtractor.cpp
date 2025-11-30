/**
 * @file FeatureExtractor.cpp
 * @brief Implementation of PE feature extraction for DRLHSS
 */

#include "Detection/AV/FeatureExtractor.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace drlhss {
namespace detection {
namespace av {

std::vector<float> PEFeatures::getAllFeatures() const {
    std::vector<float> all_features;
    all_features.reserve(2381);
    
    all_features.insert(all_features.end(), byte_histogram.begin(), byte_histogram.end());
    all_features.insert(all_features.end(), byte_entropy_histogram.begin(), byte_entropy_histogram.end());
    all_features.insert(all_features.end(), string_features.begin(), string_features.end());
    all_features.insert(all_features.end(), general_info.begin(), general_info.end());
    all_features.insert(all_features.end(), header_features.begin(), header_features.end());
    all_features.insert(all_features.end(), section_features.begin(), section_features.end());
    all_features.insert(all_features.end(), import_features.begin(), import_features.end());
    all_features.insert(all_features.end(), export_features.begin(), export_features.end());
    all_features.insert(all_features.end(), data_directory_features.begin(), data_directory_features.end());
    
    return all_features;
}

FeatureExtractor::FeatureExtractor() {
}

FeatureExtractor::~FeatureExtractor() {
}

std::vector<float> FeatureExtractor::extract(const std::string& file_path) {
    // Read file
    auto data = readFile(file_path);
    if (data.empty()) {
        last_error_ = "Failed to read file";
        return {};
    }
    
    // Validate PE
    if (!parsePEHeader(data)) {
        last_error_ = "Not a valid PE file";
        return {};
    }
    
    // Extract all features
    std::vector<float> features;
    features.reserve(2381);
    
    auto byte_hist = extractByteHistogram(data);
    auto entropy_hist = extractByteEntropyHistogram(data);
    auto string_feat = extractStringFeatures(data);
    auto general = extractGeneralInfo(data);
    auto header = extractHeaderFeatures(data);
    auto sections = extractSectionFeatures(data);
    auto imports = extractImportFeatures(data);
    auto exports = extractExportFeatures(data);
    auto data_dir = extractDataDirectoryFeatures(data);
    
    features.insert(features.end(), byte_hist.begin(), byte_hist.end());
    features.insert(features.end(), entropy_hist.begin(), entropy_hist.end());
    features.insert(features.end(), string_feat.begin(), string_feat.end());
    features.insert(features.end(), general.begin(), general.end());
    features.insert(features.end(), header.begin(), header.end());
    features.insert(features.end(), sections.begin(), sections.end());
    features.insert(features.end(), imports.begin(), imports.end());
    features.insert(features.end(), exports.begin(), exports.end());
    features.insert(features.end(), data_dir.begin(), data_dir.end());
    
    if (features.size() != 2381) {
        last_error_ = "Feature count mismatch: " + std::to_string(features.size());
        return {};
    }
    
    return features;
}

PEFeatures FeatureExtractor::extractStructured(const std::string& file_path) {
    PEFeatures pe_features;
    
    auto data = readFile(file_path);
    if (data.empty()) {
        return pe_features;
    }
    
    pe_features.byte_histogram = extractByteHistogram(data);
    pe_features.byte_entropy_histogram = extractByteEntropyHistogram(data);
    pe_features.string_features = extractStringFeatures(data);
    pe_features.general_info = extractGeneralInfo(data);
    pe_features.header_features = extractHeaderFeatures(data);
    pe_features.section_features = extractSectionFeatures(data);
    pe_features.import_features = extractImportFeatures(data);
    pe_features.export_features = extractExportFeatures(data);
    pe_features.data_directory_features = extractDataDirectoryFeatures(data);
    
    return pe_features;
}

bool FeatureExtractor::isValidPE(const std::string& file_path) {
    auto data = readFile(file_path);
    return parsePEHeader(data);
}

// Private methods

std::vector<uint8_t> FeatureExtractor::readFile(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return {};
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        return {};
    }
    
    return buffer;
}

float FeatureExtractor::calculateEntropy(const std::vector<uint8_t>& data) {
    if (data.empty()) return 0.0f;
    
    std::vector<int> freq(256, 0);
    for (auto byte : data) {
        freq[byte]++;
    }
    
    float entropy = 0.0f;
    float size = static_cast<float>(data.size());
    
    for (int count : freq) {
        if (count > 0) {
            float prob = count / size;
            entropy -= prob * std::log2(prob);
        }
    }
    
    return entropy;
}

bool FeatureExtractor::parsePEHeader(const std::vector<uint8_t>& data) {
    if (data.size() < 64) return false;
    
    // Check DOS signature
    if (data[0] != 'M' || data[1] != 'Z') {
        return false;
    }
    
    // This is a simplified check
    // Full PE parsing would be more complex
    return true;
}

std::vector<float> FeatureExtractor::extractByteHistogram(const std::vector<uint8_t>& data) {
    std::vector<float> histogram(256, 0.0f);
    
    for (auto byte : data) {
        histogram[byte] += 1.0f;
    }
    
    // Normalize
    float total = static_cast<float>(data.size());
    for (auto& val : histogram) {
        val /= total;
    }
    
    return histogram;
}

std::vector<float> FeatureExtractor::extractByteEntropyHistogram(const std::vector<uint8_t>& data) {
    // Simplified: return zeros for now
    return std::vector<float>(256, 0.0f);
}

std::vector<float> FeatureExtractor::extractStringFeatures(const std::vector<uint8_t>& data) {
    // Simplified: return zeros for now
    return std::vector<float>(104, 0.0f);
}

std::vector<float> FeatureExtractor::extractGeneralInfo(const std::vector<uint8_t>& data) {
    std::vector<float> features(10, 0.0f);
    features[0] = static_cast<float>(data.size());
    features[1] = calculateEntropy(data);
    return features;
}

std::vector<float> FeatureExtractor::extractHeaderFeatures(const std::vector<uint8_t>& data) {
    // Simplified: return zeros for now
    return std::vector<float>(62, 0.0f);
}

std::vector<float> FeatureExtractor::extractSectionFeatures(const std::vector<uint8_t>& data) {
    // Simplified: return zeros for now
    return std::vector<float>(255, 0.0f);
}

std::vector<float> FeatureExtractor::extractImportFeatures(const std::vector<uint8_t>& data) {
    // Simplified: return zeros for now
    return std::vector<float>(1280, 0.0f);
}

std::vector<float> FeatureExtractor::extractExportFeatures(const std::vector<uint8_t>& data) {
    // Simplified: return zeros for now
    return std::vector<float>(128, 0.0f);
}

std::vector<float> FeatureExtractor::extractDataDirectoryFeatures(const std::vector<uint8_t>& data) {
    // Simplified: return zeros for now
    return std::vector<float>(30, 0.0f);
}

} // namespace av
} // namespace detection
} // namespace drlhss
