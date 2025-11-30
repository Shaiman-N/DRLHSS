/**
 * @file ActionExecutor.hpp
 * @brief Action Executor for DIREWOLF
 * 
 * Executes security actions approved by Alpha through the permission system.
 * Integrates with AV, NIDPS, Sandbox, and system components.
 */

#pragma once

#include "XAITypes.hpp"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <mutex>
#include <map>

namespace xai {

/**
 * @brief Action execution result
 */
struct ActionResult {
    bool success = false;
    std::string action_id;
    std::string action_type;
    std::string message;
    std::chrono::system_clock::time_point execution_time;
    double execution_duration_ms = 0.0;
    
    // Rollback information
    bool can_rollback = false;
    std::string rollback_data;
};

/**
 * @brief Action execution callback
 */
using ActionCallback = std::function<void(const ActionResult&)>;

/**
 * @brief Action Executor
 * 
 * Executes approved security actions with logging, rollback capability,
 * and integration with all DRLHSS components.
 */
class ActionExecutor {
public:
    /**
     * @brief Constructor
     */
    ActionExecutor();
    
    /**
     * @brief Destructor
     */
    ~ActionExecutor();
    
    /**
     * @brief Initialize executor
     * @return True if successful
     */
    bool initialize();
    
    // ========== Network Actions (via NIDPS) ==========
    
    /**
     * @brief Block IP address
     * @param ip_address IP address to block
     * @param duration_seconds Block duration (0 = permanent)
     * @param reason Reason for blocking
     * @return Action result
     */
    ActionResult blockIP(
        const std::string& ip_address,
        int duration_seconds = 0,
        const std::string& reason = ""
    );
    
    /**
     * @brief Unblock IP address
     * @param ip_address IP address to unblock
     * @return Action result
     */
    ActionResult unblockIP(const std::string& ip_address);
    
    /**
     * @brief Block port
     * @param port Port number to block
     * @param protocol Protocol (TCP/UDP)
     * @param reason Reason for blocking
     * @return Action result
     */
    ActionResult blockPort(
        int port,
        const std::string& protocol = "TCP",
        const std::string& reason = ""
    );
    
    /**
     * @brief Isolate system from network
     * @param system_id System identifier
     * @return Action result
     */
    ActionResult isolateSystem(const std::string& system_id);
    
    // ========== File Actions (via AV) ==========
    
    /**
     * @brief Quarantine file
     * @param file_path Path to file
     * @param threat_info Threat information
     * @return Action result
     */
    ActionResult quarantineFile(
        const std::string& file_path,
        const ThreatInfo& threat_info
    );
    
    /**
     * @brief Delete file
     * @param file_path Path to file
     * @param secure_delete Use secure deletion
     * @return Action result
     */
    ActionResult deleteFile(
        const std::string& file_path,
        bool secure_delete = true
    );
    
    /**
     * @brief Restore file from quarantine
     * @param quarantine_id Quarantine ID
     * @param restore_path Path to restore to
     * @return Action result
     */
    ActionResult restoreFile(
        const std::string& quarantine_id,
        const std::string& restore_path
    );
    
    // ========== Process Actions ==========
    
    /**
     * @brief Terminate process
     * @param pid Process ID
     * @param force Force termination
     * @return Action result
     */
    ActionResult terminateProcess(int pid, bool force = false);
    
    /**
     * @brief Suspend process
     * @param pid Process ID
     * @return Action result
     */
    ActionResult suspendProcess(int pid);
    
    /**
     * @brief Resume process
     * @param pid Process ID
     * @return Action result
     */
    ActionResult resumeProcess(int pid);
    
    // ========== System Actions ==========
    
    /**
     * @brief Deploy patch
     * @param patch_id Patch identifier
     * @param target_system Target system
     * @return Action result
     */
    ActionResult deployPatch(
        const std::string& patch_id,
        const std::string& target_system = "localhost"
    );
    
    /**
     * @brief Update signatures
     * @param component Component to update (AV, NIDPS, etc.)
     * @return Action result
     */
    ActionResult updateSignatures(const std::string& component);
    
    /**
     * @brief Restart service
     * @param service_name Service name
     * @return Action result
     */
    ActionResult restartService(const std::string& service_name);
    
    // ========== Rollback Actions ==========
    
    /**
     * @brief Rollback action
     * @param action_id Action ID to rollback
     * @return Action result
     */
    ActionResult rollbackAction(const std::string& action_id);
    
    /**
     * @brief Check if action can be rolled back
     * @param action_id Action ID
     * @return True if rollback is possible
     */
    bool canRollback(const std::string& action_id);
    
    // ========== Action History ==========
    
    /**
     * @brief Get action history
     * @param limit Maximum number of actions
     * @return Vector of action results
     */
    std::vector<ActionResult> getActionHistory(int limit = 100);
    
    /**
     * @brief Get action by ID
     * @param action_id Action ID
     * @return Action result (empty if not found)
     */
    ActionResult getAction(const std::string& action_id);
    
    // ========== Callbacks ==========
    
    /**
     * @brief Register callback for action completion
     * @param callback Callback function
     */
    void registerActionCallback(ActionCallback callback);
    
    // ========== Statistics ==========
    
    struct ExecutorStats {
        uint64_t total_actions = 0;
        uint64_t successful_actions = 0;
        uint64_t failed_actions = 0;
        uint64_t rollbacks = 0;
        double avg_execution_time_ms = 0.0;
        std::map<std::string, uint64_t> actions_by_type;
    };
    
    ExecutorStats getStats() const;

private:
    // Action history
    std::vector<ActionResult> action_history_;
    mutable std::mutex history_mutex_;
    
    // Callbacks
    std::vector<ActionCallback> callbacks_;
    mutable std::mutex callback_mutex_;
    
    // Statistics
    mutable std::atomic<uint64_t> total_actions_{0};
    mutable std::atomic<uint64_t> successful_actions_{0};
    mutable std::atomic<uint64_t> failed_actions_{0};
    mutable std::atomic<uint64_t> rollbacks_{0};
    
    // Helper methods
    std::string generateActionId();
    void logAction(const ActionResult& result);
    void notifyCallbacks(const ActionResult& result);
    void addToHistory(const ActionResult& result);
    
    // Platform-specific implementations
    #ifdef _WIN32
    ActionResult blockIPWindows(const std::string& ip_address, int duration_seconds);
    ActionResult terminateProcessWindows(int pid, bool force);
    #else
    ActionResult blockIPLinux(const std::string& ip_address, int duration_seconds);
    ActionResult terminateProcessLinux(int pid, bool force);
    #endif
    
    // Component integration
    ActionResult executeAVAction(const std::string& action, const std::string& target);
    ActionResult executeNIDPSAction(const std::string& action, const std::string& target);
    ActionResult executeSystemAction(const std::string& action, const std::string& target);
};

} // namespace xai
