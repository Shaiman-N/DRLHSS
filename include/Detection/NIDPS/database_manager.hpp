#ifndef DATABASE_MANAGER_HPP
#define DATABASE_MANAGER_HPP

#include "packet_data.hpp"
#include "drl_framework.hpp"
#include <sqlite3.h>
#include <string>
#include <mutex>
#include <vector>

namespace nidps {

class DatabaseManager {
public:
    DatabaseManager(const std::string& db_path);
    ~DatabaseManager();
    
    bool initialize();
    bool storeAttackPattern(const AttackPattern& pattern);
    bool storePacketLog(const PacketPtr& packet);
    
    std::vector<AttackPattern> retrieveAttackPatterns(int limit = 1000);
    std::vector<AttackPattern> retrievePatternsByType(const std::string& attack_type);
    
    bool updateStatistics(const std::string& stat_type, int value);
    void cleanup(int days_to_keep = 30);
    
private:
    bool createTables();
    bool executeQuery(const std::string& query);
    
    std::string db_path_;
    sqlite3* db_;
    std::mutex db_mutex_;
    bool initialized_;
};

} // namespace nidps

#endif // DATABASE_MANAGER_HPP
