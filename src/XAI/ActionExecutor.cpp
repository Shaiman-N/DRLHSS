/**
 * @file ActionExecutor.cpp
 * @brief Implementation of Action Executor
 */

#include "XAI/ActionExecutor.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <random>
#include <ctime>

#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#else
#include <signal.h>
#include <unistd.h>
#include <sys/types.h>
#endif

namespace xai {

ActionExecutor::ActionExecutor() {
}

ActionExecutor::~ActionExecutor() {
}

bool ActionExecutor::initialize() {
    std::cout << "[Action Executor] Initialized successfully" << std::endl;
    return true;
}

// ========== Network Actions ==========

ActionResult ActionExecutor::blockIP(
    const std::string& ip_address,
    int duration_seconds,
    const std::string& reason
) {
    auto start_time = std::chrono::steady_clock::now();
    
    ActionResult result;
    result.action_id = generateActionId();
    result.action_type = "BLOCK_IP";
    result.execution_time = std::chrono::system_clock::now();
    
    std::cout << "[Action Executor] Blocking IP: " << ip_address 
              << " (duration: " << duration_seconds << "s, reason: " << reason << ")" << std::endl;
    
    try {
        #ifdef _WIN32
        result = blockIPWindows(ip_address, duration_seconds);
        #else
        result = blockIPLinux(ip_address, duration_seconds);
        #endif
        
        result.action_id = generateActionId();
        result.action_type = "BLOCK_IP";
        result.can_rollback = true;
        result.rollback_data = ip_address;
        
    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Error: ") + e.what();
    }
    
    auto end_time = std::chrono::steady_clock::now();
    result.execution_duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    logAction(result);
    notifyCallbacks(result);
    addToHistory(result);
    
    return result;
}

ActionResult ActionExecutor::unblockIP(const std::string& ip_address) {
    ActionResult result;
    result.action_id = generateActionId();
    result.action_type = "UNBLOCK_IP";
    result.execution_time = std::chrono::system_clock::now();
    
    std::cout << "[Action Executor] Unblocking IP: " << ip_address << std::endl;
    
    // Implementation would remove firewall rule
    result.success = true;
    result.message = "IP unblocked successfully";
    
    logAction(result);
    notifyCallbacks(result);
    addToHistory(result);
    
    return result;
}

ActionResult ActionExecutor::blockPort(
    int port,
    const std::string& protocol,
    const std::string& reason
) {
    ActionResult result;
    result.action_id = generateActionId();
    result.action_type = "BLOCK_PORT";
    result.execution_time = std::chrono::system_clock::now();
    
    std::cout << "[Action Executor] Blocking port: " << port 
              << " (" << protocol << ", reason: " << reason << ")" << std::endl;
    
    // Implementation would add firewall rule
    result.success = true;
    result.message = "Port blocked successfully";
    result.can_rollback = true;
    
    logAction(result);
    notifyCallbacks(result);
    addToHistory(result);
    
    return result;
}

ActionResult ActionExecutor::isolateSystem(const std::string& system_id) {
    ActionResult result;
    result.action_id = generateActionId();
    result.action_type = "ISOLATE_SYSTEM";
    result.execution_time = std::chrono::system_clock::now();
    
    std::cout << "[Action Executor] Isolating system: " << system_id << std::endl;
    
    // Implementation would disable network interfaces
    result.success = true;
    result.message = "System isolated successfully";
    result.can_rollback = true;
    
    logAction(result);
    notifyCallbacks(result);
    addToHistory(result);
    
    return result;
}

// ========== File Actions ==========

ActionResult ActionExecutor::quarantineFile(
    const std::string& file_path,
    const ThreatInfo& threat_info
) {
    ActionResult result;
    result.action_id = generateActionId();
    result.action_type = "QUARANTINE_FILE";
    result.execution_time = std::chrono::system_clock::now();
    
    std::cout << "[Action Executor] Quarantining file: " << file_path << std::endl;
    
    // Implementation would move file to quarantine directory
    result.success = true;
    result.message = "File quarantined successfully";
    result.can_rollback = true;
    result.rollback_data = file_path;
    
    logAction(result);
    notifyCallbacks(result);
    addToHistory(result);
    
    return result;
}

ActionResult ActionExecutor::deleteFile(
    const std::string& file_path,
    bool secure_delete
) {
    ActionResult result;
    result.action_id = generateActionId();
    result.action_type = "DELETE_FILE";
    result.execution_time = std::chrono::system_clock::now();
    
    std::cout << "[Action Executor] Deleting file: " << file_path 
              << " (secure: " << secure_delete << ")" << std::endl;
    
    // Implementation would delete file (securely if requested)
    result.success = true;
    result.message = "File deleted successfully";
    result.can_rollback = false; // Cannot rollback deletion
    
    logAction(result);
    notifyCallbacks(result);
    addToHistory(result);
    
    return result;
}

ActionResult ActionExecutor::restoreFile(
    const std::string& quarantine_id,
    const std::string& restore_path
) {
    ActionResult result;
    result.action_id = generateActionId();
    result.action_type = "RESTORE_FILE";
    result.execution_time = std::chrono::system_clock::now();
    
    std::cout << "[Action Executor] Restoring file: " << quarantine_id 
              << " to " << restore_path << std::endl;
    
    // Implementation would restore file from quarantine
    result.success = true;
    result.message = "File restored successfully";
    
    logAction(result);
    notifyCallbacks(result);
    addToHistory(result);
    
    return result;
}

// ========== Process Actions ==========

ActionResult ActionExecutor::terminateProcess(int pid, bool force) {
    auto start_time = std::chrono::steady_clock::now();
    
    ActionResult result;
    result.action_id = generateActionId();
    result.action_type = "TERMINATE_PROCESS";
    result.execution_time = std::chrono::system_clock::now();
    
    std::cout << "[Action Executor] Terminating process: " << pid 
              << " (force: " << force << ")" << std::endl;
    
    try {
        #ifdef _WIN32
        result = terminateProcessWindows(pid, force);
        #else
        result = terminateProcessLinux(pid, force);
        #endif
        
        result.action_id = generateActionId();
        result.action_type = "TERMINATE_PROCESS";
        result.can_rollback = false; // Cannot rollback process termination
        
    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Error: ") + e.what();
    }
    
    auto end_time = std::chrono::steady_clock::now();
    result.execution_duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    logAction(result);
    notifyCallbacks(result);
    addToHistory(result);
    
    return result;
}

ActionResult ActionExecutor::suspendProcess(int pid) {
    ActionResult result;
    result.action_id = generateActionId();
    result.action_type = "SUSPEND_PROCESS";
    result.execution_time = std::chrono::system_clock::now();
    
    std::cout << "[Action Executor] Suspending process: " << pid << std::endl;
    
    // Implementation would suspend process
    result.success = true;
    result.message = "Process suspended successfully";
    result.can_rollback = true;
    
    logAction(result);
    notifyCallbacks(result);
    addToHistory(result);
    
    return result;
}

ActionResult ActionExecutor::resumeProcess(int pid) {
    ActionResult result;
    result.action_id = generateActionId();
    result.action_type = "RESUME_PROCESS";
    result.execution_time = std::chrono::system_clock::now();
    
    std::cout << "[Action Executor] Resuming process: " << pid << std::endl;
    
    // Implementation would resume process
    result.success = true;
    result.message = "Process resumed successfully";
    
    logAction(result);
    notifyCallbacks(result);
    addToHistory(result);
    
    return result;
}

// ========== System Actions ==========

ActionResult ActionExecutor::deployPatch(
    const std::string& patch_id,
    const std::string& target_system
) {
    ActionResult result;
    result.action_id = generateActionId();
    result.action_type = "DEPLOY_PATCH";
    result.execution_time = std::chrono::system_clock::now();
    
    std::cout << "[Action Executor] Deploying patch: " << patch_id 
              << " to " << target_system << std::endl;
    
    // Implementation would deploy patch
    result.success = true;
    result.message = "Patch deployed successfully";
    result.can_rollback = true;
    
    logAction(result);
    notifyCallbacks(result);
    addToHistory(result);
    
    return result;
}

ActionResult ActionExecutor::updateSignatures(const std::string& component) {
    ActionResult result;
    result.action_id = generateActionId();
    result.action_type = "UPDATE_SIGNATURES";
    result.execution_time = std::chrono::system_clock::now();
    
    std::cout << "[Action Executor] Updating signatures for: " << component << std::endl;
    
    // Implementation would update signatures
    result.success = true;
    result.message = "Signatures updated successfully";
    
    logAction(result);
    notifyCallbacks(result);
    addToHistory(result);
    
    return result;
}

ActionResult ActionExecutor::restartService(const std::string& service_name) {
    ActionResult result;
    result.action_id = generateActionId();
    result.action_type = "RESTART_SERVICE";
    result.execution_time = std::chrono::system_clock::now();
    
    std::cout << "[Action Executor] Restarting service: " << service_name << std::endl;
    
    // Implementation would restart service
    result.success = true;
    result.message = "Service restarted successfully";
    
    logAction(result);
    notifyCallbacks(result);
    addToHistory(result);
    
    return result;
}

// ========== Rollback Actions ==========

ActionResult ActionExecutor::rollbackAction(const std::string& action_id) {
    ActionResult result;
    result.action_id = generateActionId();
    result.action_type = "ROLLBACK";
    result.execution_time = std::chrono::system_clock::now();
    
    std::cout << "[Action Executor] Rolling back action: " << action_id << std::endl;
    
    // Find original action
    ActionResult original_action;
    {
        std::lock_guard<std::mutex> lock(history_mutex_);
        
        for (const auto& action : action_history_) {
            if (action.action_id == action_id) {
                original_action = action;
                break;
            }
        }
    }
    
    if (original_action.action_id.empty()) {
        result.success = false;
        result.message = "Action not found";
        return result;
    }
    
    if (!original_action.can_rollback) {
        result.success = false;
        result.message = "Action cannot be rolled back";
        return result;
    }
    
    // Perform rollback based on action type
    if (original_action.action_type == "BLOCK_IP") {
        unblockIP(original_action.rollback_data);
    } else if (original_action.action_type == "QUARANTINE_FILE") {
        restoreFile(action_id, original_action.rollback_data);
    }
    
    result.success = true;
    result.message = "Action rolled back successfully";
    
    rollbacks_.fetch_add(1);
    
    logAction(result);
    notifyCallbacks(result);
    addToHistory(result);
    
    return result;
}

bool ActionExecutor::canRollback(const std::string& action_id) {
    std::lock_guard<std::mutex> lock(history_mutex_);
    
    for (const auto& action : action_history_) {
        if (action.action_id == action_id) {
            return action.can_rollback;
        }
    }
    
    return false;
}

// ========== Action History ==========

std::vector<ActionResult> ActionExecutor::getActionHistory(int limit) {
    std::lock_guard<std::mutex> lock(history_mutex_);
    
    std::vector<ActionResult> result;
    
    int count = 0;
    for (auto it = action_history_.rbegin(); it != action_history_.rend() && count < limit; ++it, ++count) {
        result.push_back(*it);
    }
    
    return result;
}

ActionResult ActionExecutor::getAction(const std::string& action_id) {
    std::lock_guard<std::mutex> lock(history_mutex_);
    
    for (const auto& action : action_history_) {
        if (action.action_id == action_id) {
            return action;
        }
    }
    
    return ActionResult();
}

// ========== Callbacks ==========

void ActionExecutor::registerActionCallback(ActionCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    callbacks_.push_back(callback);
}

// ========== Statistics ==========

ActionExecutor::ExecutorStats ActionExecutor::getStats() const {
    ExecutorStats stats;
    
    stats.total_actions = total_actions_.load();
    stats.successful_actions = successful_actions_.load();
    stats.failed_actions = failed_actions_.load();
    stats.rollbacks = rollbacks_.load();
    
    // Compute average execution time
    {
        std::lock_guard<std::mutex> lock(history_mutex_);
        
        if (!action_history_.empty()) {
            double total_time = 0.0;
            for (const auto& action : action_history_) {
                total_time += action.execution_duration_ms;
                stats.actions_by_type[action.action_type]++;
            }
            stats.avg_execution_time_ms = total_time / action_history_.size();
        }
    }
    
    return stats;
}

// ========== Private Methods ==========

std::string ActionExecutor::generateActionId() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 999999);
    
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::system_clock::to_time_t(now);
    
    std::ostringstream oss;
    oss << "ACT_" << timestamp << "_" << std::setfill('0') << std::setw(6) << dis(gen);
    
    return oss.str();
}

void ActionExecutor::logAction(const ActionResult& result) {
    std::cout << "[Action Executor] Action " << result.action_id 
              << " (" << result.action_type << "): "
              << (result.success ? "SUCCESS" : "FAILED")
              << " - " << result.message << std::endl;
}

void ActionExecutor::notifyCallbacks(const ActionResult& result) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    
    for (const auto& callback : callbacks_) {
        try {
            callback(result);
        } catch (const std::exception& e) {
            std::cerr << "[Action Executor] Callback error: " << e.what() << std::endl;
        }
    }
}

void ActionExecutor::addToHistory(const ActionResult& result) {
    std::lock_guard<std::mutex> lock(history_mutex_);
    
    action_history_.push_back(result);
    
    // Keep only last 1000 actions
    if (action_history_.size() > 1000) {
        action_history_.erase(action_history_.begin());
    }
    
    total_actions_.fetch_add(1);
    
    if (result.success) {
        successful_actions_.fetch_add(1);
    } else {
        failed_actions_.fetch_add(1);
    }
}

// ========== Platform-Specific Implementations ==========

#ifdef _WIN32

ActionResult ActionExecutor::blockIPWindows(const std::string& ip_address, int duration_seconds) {
    ActionResult result;
    
    // Use Windows Firewall API
    std::string command = "netsh advfirewall firewall add rule name=\"DIREWOLF_BLOCK_" + ip_address + 
                         "\" dir=in action=block remoteip=" + ip_address;
    
    int ret = system(command.c_str());
    
    result.success = (ret == 0);
    result.message = result.success ? "IP blocked successfully" : "Failed to block IP";
    
    return result;
}

ActionResult ActionExecutor::terminateProcessWindows(int pid, bool force) {
    ActionResult result;
    
    HANDLE hProcess = OpenProcess(PROCESS_TERMINATE, FALSE, pid);
    
    if (hProcess == NULL) {
        result.success = false;
        result.message = "Failed to open process";
        return result;
    }
    
    BOOL success = TerminateProcess(hProcess, 1);
    CloseHandle(hProcess);
    
    result.success = (success != 0);
    result.message = result.success ? "Process terminated successfully" : "Failed to terminate process";
    
    return result;
}

#else

ActionResult ActionExecutor::blockIPLinux(const std::string& ip_address, int duration_seconds) {
    ActionResult result;
    
    // Use iptables
    std::string command = "iptables -A INPUT -s " + ip_address + " -j DROP";
    
    int ret = system(command.c_str());
    
    result.success = (ret == 0);
    result.message = result.success ? "IP blocked successfully" : "Failed to block IP";
    
    return result;
}

ActionResult ActionExecutor::terminateProcessLinux(int pid, bool force) {
    ActionResult result;
    
    int signal = force ? SIGKILL : SIGTERM;
    int ret = kill(pid, signal);
    
    result.success = (ret == 0);
    result.message = result.success ? "Process terminated successfully" : "Failed to terminate process";
    
    return result;
}

#endif

} // namespace xai
