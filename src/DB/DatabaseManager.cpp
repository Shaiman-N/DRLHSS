#include "DB/DatabaseManager.hpp"
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <fstream>

namespace db {

DatabaseManager::DatabaseManager(const std::string& db_path)
    : db_path_(db_path), db_(nullptr) {
}

DatabaseManager::~DatabaseManager() {
    if (db_) {
        sqlite3_close(db_);
    }
}

bool DatabaseManager::initialize() {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    int rc = sqlite3_open(db_path_.c_str(), &db_);
    if (rc != SQLITE_OK) {
        std::cerr << "[DatabaseManager] Failed to open database: " << sqlite3_errmsg(db_) << std::endl;
        return false;
    }
    
    // Enable WAL mode for better concurrency
    char* err_msg = nullptr;
    rc = sqlite3_exec(db_, "PRAGMA journal_mode=WAL;", nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        std::cerr << "[DatabaseManager] Failed to enable WAL: " << err_msg << std::endl;
        sqlite3_free(err_msg);
    }
    
    if (!createTables()) {
        return false;
    }
    
    if (!createIndices()) {
        return false;
    }
    
    std::cout << "[DatabaseManager] Initialized database: " << db_path_ << std::endl;
    return true;
}

bool DatabaseManager::createTables() {
    const char* telemetry_table = R"(
        CREATE TABLE IF NOT EXISTS telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sandbox_id TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            syscall_count INTEGER,
            file_read_count INTEGER,
            file_write_count INTEGER,
            file_delete_count INTEGER,
            network_connections INTEGER,
            bytes_sent INTEGER,
            bytes_received INTEGER,
            child_processes INTEGER,
            cpu_usage REAL,
            memory_usage REAL,
            registry_modification INTEGER,
            privilege_escalation_attempt INTEGER,
            code_injection_detected INTEGER,
            artifact_hash TEXT,
            artifact_type TEXT
        );
    )";
    
    const char* experience_table = R"(
        CREATE TABLE IF NOT EXISTS experiences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id TEXT NOT NULL,
            state_vector TEXT NOT NULL,
            action INTEGER NOT NULL,
            reward REAL NOT NULL,
            next_state_vector TEXT NOT NULL,
            done INTEGER NOT NULL,
            timestamp INTEGER NOT NULL
        );
    )";
    
    const char* attack_pattern_table = R"(
        CREATE TABLE IF NOT EXISTS attack_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telemetry_features TEXT NOT NULL,
            action_taken INTEGER NOT NULL,
            reward REAL NOT NULL,
            attack_type TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            timestamp INTEGER NOT NULL,
            sandbox_id TEXT,
            artifact_hash TEXT
        );
    )";
    
    const char* model_metadata_table = R"(
        CREATE TABLE IF NOT EXISTS model_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version TEXT NOT NULL,
            training_date INTEGER NOT NULL,
            training_episodes INTEGER,
            final_average_reward REAL,
            final_loss REAL,
            learning_rate REAL,
            gamma REAL,
            epsilon_start REAL,
            epsilon_end REAL,
            batch_size INTEGER,
            target_update_frequency INTEGER,
            detection_accuracy REAL,
            false_positive_rate REAL,
            false_negative_rate REAL,
            input_dim INTEGER,
            output_dim INTEGER,
            hidden_layers TEXT
        );
    )";
    
    char* err_msg = nullptr;
    
    if (sqlite3_exec(db_, telemetry_table, nullptr, nullptr, &err_msg) != SQLITE_OK) {
        std::cerr << "[DatabaseManager] Failed to create telemetry table: " << err_msg << std::endl;
        sqlite3_free(err_msg);
        return false;
    }
    
    if (sqlite3_exec(db_, experience_table, nullptr, nullptr, &err_msg) != SQLITE_OK) {
        std::cerr << "[DatabaseManager] Failed to create experiences table: " << err_msg << std::endl;
        sqlite3_free(err_msg);
        return false;
    }
    
    if (sqlite3_exec(db_, attack_pattern_table, nullptr, nullptr, &err_msg) != SQLITE_OK) {
        std::cerr << "[DatabaseManager] Failed to create attack_patterns table: " << err_msg << std::endl;
        sqlite3_free(err_msg);
        return false;
    }
    
    if (sqlite3_exec(db_, model_metadata_table, nullptr, nullptr, &err_msg) != SQLITE_OK) {
        std::cerr << "[DatabaseManager] Failed to create model_metadata table: " << err_msg << std::endl;
        sqlite3_free(err_msg);
        return false;
    }
    
    return true;
}

bool DatabaseManager::createIndices() {
    const char* indices[] = {
        "CREATE INDEX IF NOT EXISTS idx_telemetry_sandbox ON telemetry(sandbox_id);",
        "CREATE INDEX IF NOT EXISTS idx_telemetry_hash ON telemetry(artifact_hash);",
        "CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry(timestamp);",
        "CREATE INDEX IF NOT EXISTS idx_experiences_episode ON experiences(episode_id);",
        "CREATE INDEX IF NOT EXISTS idx_patterns_type ON attack_patterns(attack_type);",
        "CREATE INDEX IF NOT EXISTS idx_patterns_timestamp ON attack_patterns(timestamp);",
        "CREATE INDEX IF NOT EXISTS idx_model_version ON model_metadata(model_version);"
    };
    
    char* err_msg = nullptr;
    for (const char* index_sql : indices) {
        if (sqlite3_exec(db_, index_sql, nullptr, nullptr, &err_msg) != SQLITE_OK) {
            std::cerr << "[DatabaseManager] Failed to create index: " << err_msg << std::endl;
            sqlite3_free(err_msg);
            return false;
        }
    }
    
    return true;
}

bool DatabaseManager::storeTelemetry(const drl::TelemetryData& telemetry) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        INSERT INTO telemetry (sandbox_id, timestamp, syscall_count, file_read_count,
                              file_write_count, file_delete_count, network_connections,
                              bytes_sent, bytes_received, child_processes, cpu_usage,
                              memory_usage, registry_modification, privilege_escalation_attempt,
                              code_injection_detected, artifact_hash, artifact_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    )";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        std::cerr << "[DatabaseManager] Failed to prepare telemetry insert: " << sqlite3_errmsg(db_) << std::endl;
        return false;
    }
    
    int64_t timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        telemetry.timestamp.time_since_epoch()).count();
    
    sqlite3_bind_text(stmt, 1, telemetry.sandbox_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 2, timestamp_ms);
    sqlite3_bind_int(stmt, 3, telemetry.syscall_count);
    sqlite3_bind_int(stmt, 4, telemetry.file_read_count);
    sqlite3_bind_int(stmt, 5, telemetry.file_write_count);
    sqlite3_bind_int(stmt, 6, telemetry.file_delete_count);
    sqlite3_bind_int(stmt, 7, telemetry.network_connections);
    sqlite3_bind_int(stmt, 8, telemetry.bytes_sent);
    sqlite3_bind_int(stmt, 9, telemetry.bytes_received);
    sqlite3_bind_int(stmt, 10, telemetry.child_processes);
    sqlite3_bind_double(stmt, 11, telemetry.cpu_usage);
    sqlite3_bind_double(stmt, 12, telemetry.memory_usage);
    sqlite3_bind_int(stmt, 13, telemetry.registry_modification ? 1 : 0);
    sqlite3_bind_int(stmt, 14, telemetry.privilege_escalation_attempt ? 1 : 0);
    sqlite3_bind_int(stmt, 15, telemetry.code_injection_detected ? 1 : 0);
    sqlite3_bind_text(stmt, 16, telemetry.artifact_hash.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 17, telemetry.artifact_type.c_str(), -1, SQLITE_TRANSIENT);
    
    int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        std::cerr << "[DatabaseManager] Failed to insert telemetry: " << sqlite3_errmsg(db_) << std::endl;
        return false;
    }
    
    return true;
}

bool DatabaseManager::storeExperience(const drl::Experience& experience, const std::string& episode_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        INSERT INTO experiences (episode_id, state_vector, action, reward, next_state_vector, done, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?);
    )";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }
    
    std::string state_str = serializeFloatVector(experience.state);
    std::string next_state_str = serializeFloatVector(experience.next_state);
    int64_t timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    sqlite3_bind_text(stmt, 1, episode_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, state_str.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 3, experience.action);
    sqlite3_bind_double(stmt, 4, experience.reward);
    sqlite3_bind_text(stmt, 5, next_state_str.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 6, experience.done ? 1 : 0);
    sqlite3_bind_int64(stmt, 7, timestamp);
    
    int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    return rc == SQLITE_DONE;
}

bool DatabaseManager::bulkStoreExperiences(const std::vector<drl::Experience>& experiences, const std::string& episode_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    sqlite3_exec(db_, "BEGIN TRANSACTION;", nullptr, nullptr, nullptr);
    
    const char* sql = R"(
        INSERT INTO experiences (episode_id, state_vector, action, reward, next_state_vector, done, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?);
    )";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
        return false;
    }
    
    for (const auto& exp : experiences) {
        std::string state_str = serializeFloatVector(exp.state);
        std::string next_state_str = serializeFloatVector(exp.next_state);
        int64_t timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        sqlite3_bind_text(stmt, 1, episode_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, state_str.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(stmt, 3, exp.action);
        sqlite3_bind_double(stmt, 4, exp.reward);
        sqlite3_bind_text(stmt, 5, next_state_str.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(stmt, 6, exp.done ? 1 : 0);
        sqlite3_bind_int64(stmt, 7, timestamp);
        
        if (sqlite3_step(stmt) != SQLITE_DONE) {
            sqlite3_finalize(stmt);
            sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
            return false;
        }
        
        sqlite3_reset(stmt);
    }
    
    sqlite3_finalize(stmt);
    sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, nullptr);
    
    return true;
}

bool DatabaseManager::storeAttackPattern(const drl::AttackPattern& pattern) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        INSERT INTO attack_patterns (telemetry_features, action_taken, reward, attack_type,
                                    confidence_score, timestamp, sandbox_id, artifact_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    )";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }
    
    std::string features_str = serializeFloatVector(pattern.telemetry_features);
    
    sqlite3_bind_text(stmt, 1, features_str.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 2, pattern.action_taken);
    sqlite3_bind_double(stmt, 3, pattern.reward);
    sqlite3_bind_text(stmt, 4, pattern.attack_type.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(stmt, 5, pattern.confidence_score);
    sqlite3_bind_int64(stmt, 6, pattern.getTimestampMs());
    sqlite3_bind_text(stmt, 7, pattern.sandbox_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 8, pattern.artifact_hash.c_str(), -1, SQLITE_TRANSIENT);
    
    int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    return rc == SQLITE_DONE;
}

bool DatabaseManager::storeModelMetadata(const drl::ModelMetadata& metadata) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = R"(
        INSERT INTO model_metadata (model_version, training_date, training_episodes,
                                   final_average_reward, final_loss, learning_rate, gamma,
                                   epsilon_start, epsilon_end, batch_size, target_update_frequency,
                                   detection_accuracy, false_positive_rate, false_negative_rate,
                                   input_dim, output_dim, hidden_layers)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    )";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }
    
    int64_t training_date_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        metadata.training_date.time_since_epoch()).count();
    
    std::stringstream hidden_layers_ss;
    for (size_t i = 0; i < metadata.hidden_layers.size(); ++i) {
        if (i > 0) hidden_layers_ss << ",";
        hidden_layers_ss << metadata.hidden_layers[i];
    }
    
    sqlite3_bind_text(stmt, 1, metadata.model_version.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 2, training_date_ms);
    sqlite3_bind_int(stmt, 3, metadata.training_episodes);
    sqlite3_bind_double(stmt, 4, metadata.final_average_reward);
    sqlite3_bind_double(stmt, 5, metadata.final_loss);
    sqlite3_bind_double(stmt, 6, metadata.learning_rate);
    sqlite3_bind_double(stmt, 7, metadata.gamma);
    sqlite3_bind_double(stmt, 8, metadata.epsilon_start);
    sqlite3_bind_double(stmt, 9, metadata.epsilon_end);
    sqlite3_bind_int(stmt, 10, metadata.batch_size);
    sqlite3_bind_int(stmt, 11, metadata.target_update_frequency);
    sqlite3_bind_double(stmt, 12, metadata.detection_accuracy);
    sqlite3_bind_double(stmt, 13, metadata.false_positive_rate);
    sqlite3_bind_double(stmt, 14, metadata.false_negative_rate);
    sqlite3_bind_int(stmt, 15, metadata.input_dim);
    sqlite3_bind_int(stmt, 16, metadata.output_dim);
    sqlite3_bind_text(stmt, 17, hidden_layers_ss.str().c_str(), -1, SQLITE_TRANSIENT);
    
    int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    return rc == SQLITE_DONE;
}

DatabaseManager::DatabaseStats DatabaseManager::getStats() {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    DatabaseStats stats;
    
    const char* queries[] = {
        "SELECT COUNT(*) FROM telemetry;",
        "SELECT COUNT(*) FROM experiences;",
        "SELECT COUNT(*) FROM attack_patterns;",
        "SELECT COUNT(*) FROM model_metadata;"
    };
    
    int64_t* counts[] = {
        &stats.telemetry_count,
        &stats.experience_count,
        &stats.pattern_count,
        &stats.model_count
    };
    
    for (size_t i = 0; i < 4; ++i) {
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_, queries[i], -1, &stmt, nullptr) == SQLITE_OK) {
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                *counts[i] = sqlite3_column_int64(stmt, 0);
            }
            sqlite3_finalize(stmt);
        }
    }
    
    // Get database file size
    std::ifstream file(db_path_, std::ios::binary | std::ios::ate);
    if (file.is_open()) {
        stats.db_size_bytes = file.tellg();
        file.close();
    }
    
    return stats;
}

bool DatabaseManager::vacuum() {
    std::lock_guard<std::mutex> lock(db_mutex_);
    char* err_msg = nullptr;
    int rc = sqlite3_exec(db_, "VACUUM;", nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        std::cerr << "[DatabaseManager] Vacuum failed: " << err_msg << std::endl;
        sqlite3_free(err_msg);
        return false;
    }
    return true;
}

bool DatabaseManager::backup(const std::string& backup_path) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    sqlite3* backup_db;
    int rc = sqlite3_open(backup_path.c_str(), &backup_db);
    if (rc != SQLITE_OK) {
        return false;
    }
    
    sqlite3_backup* backup_handle = sqlite3_backup_init(backup_db, "main", db_, "main");
    if (backup_handle) {
        sqlite3_backup_step(backup_handle, -1);
        sqlite3_backup_finish(backup_handle);
    }
    
    rc = sqlite3_errcode(backup_db);
    sqlite3_close(backup_db);
    
    return rc == SQLITE_OK;
}

std::string DatabaseManager::serializeFloatVector(const std::vector<float>& vec) {
    std::stringstream ss;
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) ss << ",";
        ss << vec[i];
    }
    return ss.str();
}

std::vector<float> DatabaseManager::deserializeFloatVector(const std::string& str) {
    std::vector<float> vec;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        vec.push_back(std::stof(token));
    }
    return vec;
}

float DatabaseManager::computeCosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 0.0f;
    
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for (size_t i = 0; i < a.size(); ++i) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
    
    return dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

} // namespace db
