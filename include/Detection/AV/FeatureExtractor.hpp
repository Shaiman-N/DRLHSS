/**
 * @file FeatureExtractor.hpp
 * @brief PE feature extraction for DRLHSS
 * 
 * Extracts 2381 PE features from executable files
 * Compatible with EMBER feature set for ML-based malware detection
 */

#ifndef DRLHSS_FEATURE_EXTRACTOR_HPP
#define DRLHSS_FEATURE_EXTRACTOR_HPP

#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace drlhss {
namespace detection {
namespace av {

/**
 * @brief PE Feature Categories (EMBER-compatible)
 */
struct PEFeatures {
    // Byte histogram (256 features)
    std::vector<float> byte_histogram;
    
    // Byte entropy histogram (256 features)
    std::vector<float> byte_entropy_histogram;
    
    // String features (104 features)
    std::vector<float> string_features;
    
    // General file info (10 features)
    std::vector<float> general_info;
    
    // Header features (62 features)
    std::vector<float> header_features;
    
    // Section features (255 features)
    std::vector<float> section_features;
    
    // Import features (1280 features)
    std::vector<float> import_features;
    
    // Export features (128 features)
    std::vector<float> export_features;
    
    // Data directory features (30 features)
    std::vector<float> data_directory_features;
    
    // Total: 2381 features
    std::vector<float> getAllFeatures() const;
};

/**
 * @brief FeatureExtractor - Extracts PE file features for ML inference
 */
class FeatureExtractor {
public:
    FeatureExtractor();
    ~FeatureExtractor();
    
    /**
     * @brief Extract all 2381 features from a PE file
     * @param file_path Path to PE file
     * @return Feature vector of size 2381, empty on error
     */
    std::vector<float> extract(const std::string& file_path);
    
    /**
     * @brief Extract features into structured format
     * @param file_path Path to PE file
     * @return PEFeatures structure
     */
    PEFeatures extractStructured(const std::string& file_path);
    
    /**
     * @brief Validate if file is a valid PE
     * @param file_path Path to file
     * @return true if valid PE file
     */
    bool isValidPE(const std::string& file_path);
    
    /**
     * @brief Get last error message
     */
    std::string getLastError() const { return last_error_; }
    
private:
    std::string last_error_;
    
    // Feature extraction methods
    std::vector<float> extractByteHistogram(const std::vector<uint8_t>& data);
    std::vector<float> extractByteEntropyHistogram(const std::vector<uint8_t>& data);
    std::vector<float> extractStringFeatures(const std::vector<uint8_t>& data);
    std::vector<float> extractGeneralInfo(const std::vector<uint8_t>& data);
    std::vector<float> extractHeaderFeatures(const std::vector<uint8_t>& data);
    std::vector<float> extractSectionFeatures(const std::vector<uint8_t>& data);
    std::vector<float> extractImportFeatures(const std::vector<uint8_t>& data);
    std::vector<float> extractExportFeatures(const std::vector<uint8_t>& data);
    std::vector<float> extractDataDirectoryFeatures(const std::vector<uint8_t>& data);
    
    // Helper methods
    std::vector<uint8_t> readFile(const std::string& file_path);
    float calculateEntropy(const std::vector<uint8_t>& data);
    bool parsePEHeader(const std::vector<uint8_t>& data);
};

} // namespace av
} // namespace detection
} // namespace drlhss

#endif // DRLHSS_FEATURE_EXTRACTOR_HPP
