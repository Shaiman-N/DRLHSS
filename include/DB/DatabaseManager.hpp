#pragma once

#include "DRL/TelemetryData.hpp"
#include "DRL/Experience.hpp"
#include "DRL/AttackPattern.hpp"
#include "DRL/ModelMetadata.hpp"
#include <sqlite3.h>
#include <string>
#include <vector>
#include <memory>
#include <mutex>

namespace db {

/**
 * @brief Production-grade database manager for DRL system
 * 
 * Thread-safe SQLite wrapper for storing telemetry, experiences,
 * attack patterns, and model metadata with connection pooling.
 */
class DatabaseManager {
public:
    /**
     * @brief Constructor
     * @param db_path Path to SQLite database file
     */
    explicit DatabaseManager(const std::string& db_path);
    
    /**
     * @brief Destructor
     */
    ~DatabaseManager();
    
    /**
     * @brief Initialize database schema
     * @return True if successful
     */
    bool initialize();
    
    // Telemetry operations
    bool storeTelemetry(const drl::TelemetryData& telemetry);
    std::vector<drl::TelemetryData> queryTelemetry(const std::string& sandbox_id, int limit = 100);
    std::vector<drl::TelemetryData> queryTelemetryByHash(const std::string& artifact_hash);
    
    // Experience operations
    bool storeExperience(const drl::Experience& experience, const std::string& episode_id);
    std::vector<drl::Experience> queryExperiences(const std::string& episode_id, int limit = 1000);
    bool bulkStoreExperiences(const std::vector<drl::Experience>& experiences, const std::string& episode_id);
    
    // Attack pattern operations
    bool storeAttackPattern(const drl::AttackPattern& pattern);
    std::vector<drl::AttackPattern> queryAttackPatterns(const std::string& attack_type, int limit = 100);
    std::vector<drl::AttackPattern> querySimilarPatterns(const std::vector<float>& features, float threshold = 0.8f, int limit = 10);
    
    // Model metadata operations
    bool storeModelMetadata(const drl::ModelMetadata& metadata);
    drl::ModelMetadata queryLatestModelMetadata();
    std::vector<drl::ModelMetadata> queryModelHistory(int limit = 10);
    
    // Statistics and maintenance
    struct DatabaseStats {
        int64_t telemetry_count = 0;
        int64_t experience_count = 0;
        int64_t pattern_count = 0;
        int64_t model_count = 0;
        int64_t db_size_bytes = 0;
    };
    
    DatabaseStats getStats();
    bool vacuum();
    bool backup(const std::string& backup_path);
    
    /**
     * @brief Execute custom SQL query (for advanced use)
     * @param query SQL query string
     * @return True if successful
     */
    bool executeQuery(const std::string& query);

private:
    std::string db_path_;
    sqlite3* db_;
    mutable std::mutex db_mutex_;
    
    // Helper methods
    bool createTables();
    bool createIndices();
    std::string serializeFloatVector(const std::vector<float>& vec);
    std::vector<float> deserializeFloatVector(const std::string& str);
    float computeCosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);
};

} // namespace db
