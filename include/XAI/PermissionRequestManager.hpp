#pragma once

#include <string>
#include <map>
#include <vector>
#include <chrono>
#include <optional>
#include <functional>
#include <memory>
#include <mutex>

namespace DRLHSS {
namespace XAI {

// Forward declarations
struct ThreatEvent;
struct RecommendedAction;

/**
 * @brief Permission Request Manager
 * 
 * CRITICAL COMPONENT: Ensures Wolf NEVER takes action without Alpha's permission.
 * This is the core of Alpha's authority over the system.
 * 
 * All security actions MUST go through this manager to request permission.
 */
class PermissionRequestManager {
public:
    /**
     * @brief Permission request structure
     */
    struct PermissionRequest {
        std::string request_id;
        ThreatEvent threat;
        RecommendedAction recommendation;
        float confidence;
        std::string rationale;
        std::chrono::system_clock::time_point timestamp;
        bool is_critical;  // Critical threats get immediate attention
    };

    /**
     * @brief Alpha's response to permission request
     */
    struct PermissionResponse {
        std::string request_id;
        bool granted;  // true = permission granted, false = rejected
        std::string alpha_instruction;  // Alternative action if rejected
        std::chrono::system_clock::time_point response_time;
        std::string alpha_user_id;  // Which Alpha responded
    };

    /**
     * @brief Callback for when permission is needed
     * This triggers the voice/UI interaction with Alpha
     */
    using PermissionCallback = std::function<void(const PermissionRequest&)>;

    /**
     * @brief Constructor
     */
    PermissionRequestManager();
    ~PermissionRequestManager();

    /**
     * @brief Initialize the permission manager
     * @param callback Function to call when permission is needed
     * @return true if initialization successful
     */
    bool initialize(PermissionCallback callback);

    /**
     * @brief Request permission from Alpha
     * 
     * This is the ONLY way to get authorization for security actions.
     * Wolf will wait for Alpha's response before proceeding.
     * 
     * @param threat The detected threat
     * @param recommendation Wolf's recommended action
     * @param rationale Explanation of why this action is recommended
     * @return Request ID for tracking
     */
    std::string requestPermission(
        const ThreatEvent& threat,
        const RecommendedAction& recommendation,
        const std::string& rationale
    );

    /**
     * @brief Wait for Alpha's response (blocking with timeout)
     * 
     * @param request_id The request to wait for
     * @param timeout Maximum time to wait (default 60 seconds)
     * @return Alpha's response, or nullopt if timeout
     */
    std::optional<PermissionResponse> waitForResponse(
        const std::string& request_id,
        std::chrono::seconds timeout = std::chrono::seconds(60)
    );

    /**
     * @brief Check if response received (non-blocking)
     * 
     * @param request_id The request to check
     * @return Alpha's response if available
     */
    std::optional<PermissionResponse> checkResponse(const std::string& request_id);

    /**
     * @brief Submit Alpha's response
     * 
     * Called by the voice/UI interface when Alpha makes a decision
     * 
     * @param response Alpha's decision
     */
    void submitResponse(const PermissionResponse& response);

    /**
     * @brief Execute authorized action
     * 
     * Only executes if Alpha granted permission
     * 
     * @param response Alpha's response
     * @return true if action executed successfully
     */
    bool executeAuthorizedAction(const PermissionResponse& response);

    /**
     * @brief Record Alpha's decision for learning
     * 
     * Wolf learns from Alpha's decisions to improve future recommendations
     * 
     * @param response Alpha's decision
     */
    void recordAlphaDecision(const PermissionResponse& response);

    /**
     * @brief Get all pending requests
     * 
     * @return Vector of requests awaiting Alpha's response
     */
    std::vector<PermissionRequest> getPendingRequests() const;

    /**
     * @brief Get request by ID
     * 
     * @param request_id Request identifier
     * @return Request if found
     */
    std::optional<PermissionRequest> getRequest(const std::string& request_id) const;

    /**
     * @brief Cancel a pending request
     * 
     * @param request_id Request to cancel
     * @return true if cancelled successfully
     */
    bool cancelRequest(const std::string& request_id);

    /**
     * @brief Get Alpha's decision history
     * 
     * Used for learning Alpha's preferences
     * 
     * @param limit Maximum number of decisions to return
     * @return Recent decisions
     */
    std::vector<PermissionResponse> getDecisionHistory(size_t limit = 100) const;

    /**
     * @brief Analyze Alpha's decision patterns
     * 
     * Identifies patterns in Alpha's decisions to improve recommendations
     * 
     * @return Analysis results
     */
    std::map<std::string, float> analyzeAlphaPreferences() const;

private:
    // Pending requests awaiting Alpha's response
    std::map<std::string, PermissionRequest> pending_requests_;
    
    // Completed responses from Alpha
    std::map<std::string, PermissionResponse> responses_;
    
    // Decision history for learning
    std::vector<PermissionResponse> decision_history_;
    
    // Callback to trigger permission request UI/voice
    PermissionCallback permission_callback_;
    
    // Thread safety
    mutable std::mutex mutex_;
    
    // Generate unique request ID
    std::string generateRequestId();
    
    // Execute the actual security action
    bool executeAction(const RecommendedAction& action);
    
    // Log permission request and response
    void logPermissionRequest(const PermissionRequest& request);
    void logPermissionResponse(const PermissionResponse& response);
};

} // namespace XAI
} // namespace DRLHSS
