#include "XAI/PermissionRequestManager.hpp"
#include "XAI/XAITypes.hpp"
#include <sstream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <condition_variable>

namespace DRLHSS {
namespace XAI {

PermissionRequestManager::PermissionRequestManager() {
    // Constructor
}

PermissionRequestManager::~PermissionRequestManager() {
    // Destructor
}

bool PermissionRequestManager::initialize(PermissionCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!callback) {
        return false;
    }
    
    permission_callback_ = callback;
    return true;
}

std::string PermissionRequestManager::requestPermission(
    const ThreatEvent& threat,
    const RecommendedAction& recommendation,
    const std::string& rationale
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Generate unique request ID
    std::string request_id = generateRequestId();
    
    // Create permission request
    PermissionRequest request;
    request.request_id = request_id;
    request.threat = threat;
    request.recommendation = recommendation;
    request.confidence = threat.confidence;
    request.rationale = rationale;
    request.timestamp = std::chrono::system_clock::now();
    request.is_critical = (threat.base_event.severity == Severity::CRITICAL || 
                          threat.base_event.severity == Severity::EMERGENCY);
    
    // Store pending request
    pending_requests_[request_id] = request;
    
    // Log the request
    logPermissionRequest(request);
    
    // Trigger callback to notify UI/voice interface
    if (permission_callback_) {
        // Call callback without holding lock to avoid deadlock
        mutex_.unlock();
        permission_callback_(request);
        mutex_.lock();
    }
    
    return request_id;
}

std::optional<PermissionRequestManager::PermissionResponse> 
PermissionRequestManager::waitForResponse(
    const std::string& request_id,
    std::chrono::seconds timeout
) {
    auto start_time = std::chrono::steady_clock::now();
    
    while (true) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            
            // Check if response available
            auto it = responses_.find(request_id);
            if (it != responses_.end()) {
                return it->second;
            }
        }
        
        // Check timeout
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (elapsed >= timeout) {
            return std::nullopt;  // Timeout
        }
        
        // Sleep briefly before checking again
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

std::optional<PermissionRequestManager::PermissionResponse> 
PermissionRequestManager::checkResponse(const std::string& request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = responses_.find(request_id);
    if (it != responses_.end()) {
        return it->second;
    }
    
    return std::nullopt;
}

void PermissionRequestManager::submitResponse(const PermissionResponse& response) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Store response
    responses_[response.request_id] = response;
    
    // Remove from pending
    pending_requests_.erase(response.request_id);
    
    // Add to decision history
    decision_history_.push_back(response);
    
    // Keep history limited to last 1000 decisions
    if (decision_history_.size() > 1000) {
        decision_history_.erase(decision_history_.begin());
    }
    
    // Log the response
    logPermissionResponse(response);
}

bool PermissionRequestManager::executeAuthorizedAction(
    const PermissionResponse& response
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Verify response exists
    auto resp_it = responses_.find(response.request_id);
    if (resp_it == responses_.end()) {
        return false;
    }
    
    // Get original request
    auto req_it = pending_requests_.find(response.request_id);
    if (req_it == pending_requests_.end()) {
        // Request already processed or doesn't exist
        return false;
    }
    
    const PermissionRequest& request = req_it->second;
    
    // Check if permission was granted
    if (!response.granted) {
        // Alpha rejected the recommendation
        // Check if Alpha provided alternative instruction
        if (!response.alpha_instruction.empty()) {
            // TODO: Parse and execute Alpha's alternative instruction
            // For now, just log it
        }
        return false;
    }
    
    // Permission granted - execute the action
    bool success = executeAction(request.recommendation);
    
    return success;
}

void PermissionRequestManager::recordAlphaDecision(
    const PermissionResponse& response
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Get original request for context
    auto req_it = pending_requests_.find(response.request_id);
    if (req_it == pending_requests_.end()) {
        return;
    }
    
    const PermissionRequest& request = req_it->second;
    
    // Analyze the decision
    // TODO: Implement machine learning to identify patterns
    // For now, just store the decision
    
    // Example pattern analysis:
    // - If Alpha consistently rejects certain types of actions
    // - If Alpha prefers different actions for specific threat types
    // - If Alpha's decisions correlate with time of day, system state, etc.
}

std::vector<PermissionRequestManager::PermissionRequest> 
PermissionRequestManager::getPendingRequests() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<PermissionRequest> requests;
    requests.reserve(pending_requests_.size());
    
    for (const auto& pair : pending_requests_) {
        requests.push_back(pair.second);
    }
    
    return requests;
}

std::optional<PermissionRequestManager::PermissionRequest> 
PermissionRequestManager::getRequest(const std::string& request_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = pending_requests_.find(request_id);
    if (it != pending_requests_.end()) {
        return it->second;
    }
    
    return std::nullopt;
}

bool PermissionRequestManager::cancelRequest(const std::string& request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = pending_requests_.find(request_id);
    if (it != pending_requests_.end()) {
        pending_requests_.erase(it);
        return true;
    }
    
    return false;
}

std::vector<PermissionRequestManager::PermissionResponse> 
PermissionRequestManager::getDecisionHistory(size_t limit) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t start = 0;
    if (decision_history_.size() > limit) {
        start = decision_history_.size() - limit;
    }
    
    return std::vector<PermissionResponse>(
        decision_history_.begin() + start,
        decision_history_.end()
    );
}

std::map<std::string, float> PermissionRequestManager::analyzeAlphaPreferences() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::map<std::string, float> preferences;
    
    if (decision_history_.empty()) {
        return preferences;
    }
    
    // Analyze approval rate
    size_t approved = 0;
    for (const auto& decision : decision_history_) {
        if (decision.granted) {
            approved++;
        }
    }
    
    preferences["overall_approval_rate"] = 
        static_cast<float>(approved) / decision_history_.size();
    
    // TODO: Add more sophisticated analysis
    // - Approval rate by threat type
    // - Approval rate by action type
    // - Approval rate by time of day
    // - Approval rate by confidence level
    
    return preferences;
}

std::string PermissionRequestManager::generateRequestId() {
    // Generate unique ID using timestamp + random component
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    ).count();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999);
    
    std::ostringstream oss;
    oss << "REQ-" << timestamp << "-" << dis(gen);
    
    return oss.str();
}

bool PermissionRequestManager::executeAction(const RecommendedAction& action) {
    // Execute the actual security action
    // This will integrate with existing DRLHSS components
    
    switch (action.type) {
        case ActionType::BLOCK_IP:
            // TODO: Call NIDPS to block IP
            break;
            
        case ActionType::QUARANTINE_FILE:
            // TODO: Call AV to quarantine file
            break;
            
        case ActionType::ISOLATE_SYSTEM:
            // TODO: Call network isolation
            break;
            
        case ActionType::TERMINATE_PROCESS:
            // TODO: Call process termination
            break;
            
        case ActionType::BLOCK_NETWORK:
            // TODO: Call network blocking
            break;
            
        case ActionType::ALERT_ONLY:
            // Just log, no action needed
            break;
            
        case ActionType::DEPLOY_PATCH:
            // TODO: Call patch deployment
            break;
            
        case ActionType::ROLLBACK_CHANGES:
            // TODO: Call rollback system
            break;
            
        case ActionType::CUSTOM_ACTION:
            // TODO: Execute custom action
            break;
    }
    
    return true;
}

void PermissionRequestManager::logPermissionRequest(
    const PermissionRequest& request
) {
    // TODO: Integrate with logging system
    // For now, just output to console
    
    std::cout << "[PERMISSION REQUEST] ID: " << request.request_id << std::endl;
    std::cout << "  Threat: " << request.threat.base_event.description << std::endl;
    std::cout << "  Confidence: " << request.confidence << std::endl;
    std::cout << "  Rationale: " << request.rationale << std::endl;
    std::cout << "  Critical: " << (request.is_critical ? "YES" : "NO") << std::endl;
}

void PermissionRequestManager::logPermissionResponse(
    const PermissionResponse& response
) {
    // TODO: Integrate with logging system
    
    std::cout << "[PERMISSION RESPONSE] ID: " << response.request_id << std::endl;
    std::cout << "  Granted: " << (response.granted ? "YES" : "NO") << std::endl;
    std::cout << "  Alpha: " << response.alpha_user_id << std::endl;
    
    if (!response.granted && !response.alpha_instruction.empty()) {
        std::cout << "  Alternative: " << response.alpha_instruction << std::endl;
    }
}

} // namespace XAI
} // namespace DRLHSS
