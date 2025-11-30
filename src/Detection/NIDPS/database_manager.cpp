#include "database_manager.hpp"
#include <iostream>
#include <sstream>
#include <ctime>

namespace nidps {

DatabaseManager::DatabaseManager(const std::string& db_path)
    : db_path_(db_path), db_(nullptr), initialized_(false) {}

DatabaseManager::~DatabaseManager() {
    if (db_) {
        sqlite3_close(db_);
    }
}

bool DatabaseManager::initialize() {
    int rc = sqlite3_open(db_path_.c_str(), &db_);
    if (rc != SQLITE_OK) {
        std::cerr << "Cannot open database: " << sqlite3_errmsg(db_) << std::endl;
        return false;
    }
    
    if (!createTables()) {
        return false;
    }
    
    initialized_ = true;
    std::cout << "Database initialized: " << db_path_ << std::endl;
    return true;
}

bool DatabaseManager::createTables() {
    const char* attack_patterns_table = R"(
        CREATE TABLE IF NOT EXISTS attack_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attack_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            threat_score INTEGER NOT NULL,
            feature_vector BLOB,
            learned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_system_modified INTEGER,
            registry_modified INTEGER,
            network_activity INTEGER,
            process_created INTEGER,
            memory_injection INTEGER,
            suspicious_api_calls INTEGER
        );
    )";
    
    const char* packet_logs_table = R"(
        CREATE TABLE IF NOT EXISTS packet_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            packet_id INTEGER NOT NULL,
            source_ip TEXT,
            dest_ip TEXT,
            source_port INTEGER,
            dest_port INTEGER,
            protocol INTEGER,
            status TEXT,
            sandbox_passes INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    )";
    
    const char* statistics_table = R"(
        CREATE TABLE IF NOT EXISTS statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stat_type TEXT NOT NULL,
            stat_value INTEGER NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    )";
    
    const char* api_calls_table = R"(
        CREATE TABLE IF NOT EXISTS api_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_id INTEGER,
            api_name TEXT NOT NULL,
            FOREIGN KEY(pattern_id) REFERENCES attack_patterns(id)
        );
    )";
    
    const char* network_connections_table = R"(
        CREATE TABLE IF NOT EXISTS network_connections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_id INTEGER,
            connection TEXT NOT NULL,
            FOREIGN KEY(pattern_id) REFERENCES attack_patterns(id)
        );
    )";
    
    return executeQuery(attack_patterns_table) &&
           executeQuery(packet_logs_table) &&
           executeQuery(statistics_table) &&
           executeQuery(api_calls_table) &&
           executeQuery(network_connections_table);
}

bool DatabaseManager::executeQuery(const std::string& query) {
    char* err_msg = nullptr;
    int rc = sqlite3_exec(db_, query.c_str(), nullptr, nullptr, &err_msg);
    
    if (rc != SQLITE_OK) {
        std::cerr << "SQL error: " << err_msg << std::endl;
        sqlite3_free(err_msg);
        return false;
    }
    
    return true;
}

bool DatabaseManager::storeAttackPattern(const AttackPattern& pattern) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    std::stringstream ss;
    ss << "INSERT INTO attack_patterns (attack_type, confidence, threat_score, "
       << "file_system_modified, registry_modified, network_activity, "
       << "process_created, memory_injection, suspicious_api_calls) VALUES ("
       << "'" << pattern.attack_type << "', "
       << pattern.confidence << ", "
       << pattern.behavior.threat_score << ", "
       << pattern.behavior.file_system_modified << ", "
       << pattern.behavior.registry_modified << ", "
       << pattern.behavior.network_activity << ", "
       << pattern.behavior.process_created << ", "
       << pattern.behavior.memory_injection << ", "
       << pattern.behavior.suspicious_api_calls << ");";
    
    if (!executeQuery(ss.str())) {
        return false;
    }
    
    int64_t pattern_id = sqlite3_last_insert_rowid(db_);
    
    // Store API calls
    for (const auto& api : pattern.behavior.api_calls) {
        std::stringstream api_ss;
        api_ss << "INSERT INTO api_calls (pattern_id, api_name) VALUES ("
               << pattern_id << ", '" << api << "');";
        executeQuery(api_ss.str());
    }
    
    // Store network connections
    for (const auto& conn : pattern.behavior.network_connections) {
        std::stringstream conn_ss;
        conn_ss << "INSERT INTO network_connections (pattern_id, connection) VALUES ("
               << pattern_id << ", '" << conn << "');";
        executeQuery(conn_ss.str());
    }
    
    std::cout << "Stored attack pattern: " << pattern.attack_type 
              << " (ID: " << pattern_id << ")" << std::endl;
    
    return true;
}

bool DatabaseManager::storePacketLog(const PacketPtr& packet) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    std::string status_str;
    switch (packet->status) {
        case PacketStatus::CLEAN: status_str = "CLEAN"; break;
        case PacketStatus::SUSPICIOUS: status_str = "SUSPICIOUS"; break;
        case PacketStatus::MALICIOUS: status_str = "MALICIOUS"; break;
        case PacketStatus::CLEANED: status_str = "CLEANED"; break;
        case PacketStatus::PROCESSING: status_str = "PROCESSING"; break;
    }
    
    std::stringstream ss;
    ss << "INSERT INTO packet_logs (packet_id, source_ip, dest_ip, "
       << "source_port, dest_port, protocol, status, sandbox_passes) VALUES ("
       << packet->packet_id << ", "
       << "'" << packet->source_ip << "', "
       << "'" << packet->dest_ip << "', "
       << packet->source_port << ", "
       << packet->dest_port << ", "
       << static_cast<int>(packet->protocol) << ", "
       << "'" << status_str << "', "
       << packet->sandbox_pass_count << ");";
    
    return executeQuery(ss.str());
}

std::vector<AttackPattern> DatabaseManager::retrieveAttackPatterns(int limit) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    std::vector<AttackPattern> patterns;
    
    std::stringstream ss;
    ss << "SELECT attack_type, confidence, threat_score, "
       << "file_system_modified, registry_modified, network_activity, "
       << "process_created, memory_injection, suspicious_api_calls "
       << "FROM attack_patterns ORDER BY learned_at DESC LIMIT " << limit << ";";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, ss.str().c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            AttackPattern pattern;
            pattern.attack_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            pattern.confidence = sqlite3_column_double(stmt, 1);
            pattern.behavior.threat_score = sqlite3_column_int(stmt, 2);
            pattern.behavior.file_system_modified = sqlite3_column_int(stmt, 3);
            pattern.behavior.registry_modified = sqlite3_column_int(stmt, 4);
            pattern.behavior.network_activity = sqlite3_column_int(stmt, 5);
            pattern.behavior.process_created = sqlite3_column_int(stmt, 6);
            pattern.behavior.memory_injection = sqlite3_column_int(stmt, 7);
            pattern.behavior.suspicious_api_calls = sqlite3_column_int(stmt, 8);
            
            patterns.push_back(pattern);
        }
        sqlite3_finalize(stmt);
    }
    
    return patterns;
}

std::vector<AttackPattern> DatabaseManager::retrievePatternsByType(const std::string& attack_type) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    std::vector<AttackPattern> patterns;
    
    std::stringstream ss;
    ss << "SELECT attack_type, confidence, threat_score, "
       << "file_system_modified, registry_modified, network_activity, "
       << "process_created, memory_injection, suspicious_api_calls "
       << "FROM attack_patterns WHERE attack_type = '" << attack_type 
       << "' ORDER BY learned_at DESC;";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, ss.str().c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            AttackPattern pattern;
            pattern.attack_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            pattern.confidence = sqlite3_column_double(stmt, 1);
            pattern.behavior.threat_score = sqlite3_column_int(stmt, 2);
            pattern.behavior.file_system_modified = sqlite3_column_int(stmt, 3);
            pattern.behavior.registry_modified = sqlite3_column_int(stmt, 4);
            pattern.behavior.network_activity = sqlite3_column_int(stmt, 5);
            pattern.behavior.process_created = sqlite3_column_int(stmt, 6);
            pattern.behavior.memory_injection = sqlite3_column_int(stmt, 7);
            pattern.behavior.suspicious_api_calls = sqlite3_column_int(stmt, 8);
            
            patterns.push_back(pattern);
        }
        sqlite3_finalize(stmt);
    }
    
    return patterns;
}

bool DatabaseManager::updateStatistics(const std::string& stat_type, int value) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    std::stringstream ss;
    ss << "INSERT OR REPLACE INTO statistics (stat_type, stat_value) VALUES ("
       << "'" << stat_type << "', " << value << ");";
    
    return executeQuery(ss.str());
}

void DatabaseManager::cleanup(int days_to_keep) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    std::stringstream ss;
    ss << "DELETE FROM packet_logs WHERE timestamp < datetime('now', '-" 
       << days_to_keep << " days');";
    
    executeQuery(ss.str());
    
    std::cout << "Database cleanup completed (kept last " << days_to_keep << " days)" << std::endl;
}

} // namespace nidps
