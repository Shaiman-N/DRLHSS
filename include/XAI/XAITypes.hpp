#pragma once

#include <string>
#include <vector>
#include <map>
#include <chrono>

namespace DRLHSS {
namespace XAI {

/**
 * @brief Threat severity levels
 */
enum class Severity {
    LOW,
    MEDIUM,
    HIGH,
    CRITICAL,
    EMERGENCY
};

/**
 * @brief Event types
 */
enum class EventType {
    MALWARE_DETECTED,
    INTRUSION_ATTEMPT,
    ANOMALY_DETECTED,
    POLICY_VIOLATION,
    SUSPICIOUS_BEHAVIOR,
    ZERO_DAY_EXPLOIT,
    DATA_EXFILTRATION,
    LATERAL_MOVEMENT,
    PRIVILEGE_ESCALATION,
    RANSOMWARE_ACTIVITY
};

/**
 * @brief Recommended actions
 */
enum class ActionType {
    BLOCK_IP,
    QUARANTINE_FILE,
    ISOLATE_SYSTEM,
    TERMINATE_PROCESS,
    BLOCK_NETWORK,
    ALERT_ONLY,
    DEPLOY_PATCH,
    ROLLBACK_CHANGES,
    CUSTOM_ACTION
};

/**
 * @brief Urgency levels for Wolf's communication
 */
enum class UrgencyLevel {
    ROUTINE,      // Normal conversation tone
    ELEVATED,     // Increased alertness
    CRITICAL,     // Urgent but controlled
    EMERGENCY     // Maximum urgency
};

/**
 * @brief Security event structure
 */
struct SecurityEvent {
    std::string event_id;
    std::chrono::system_clock::time_point timestamp;
    EventType type;
    Severity severity;
    std::string source_component;  // "AV", "NIDPS", "DRL", etc.
    std::map<std::string, std::string> metadata;
    std::vector<std::string> affected_systems;
    std::string description;
};

/**
 * @brief Feature attribution for explainability
 */
struct FeatureAttribution {
    std::string feature_name;
    float importance;  // 0.0 to 1.0
    std::string description;
    std::string value;  // Actual value of the feature
};

/**
 * @brief Recommended action structure
 */
struct RecommendedAction {
    ActionType type;
    std::string description;
    std::vector<std::string> targets;  // IPs, files, processes, etc.
    std::map<std::string, std::string> parameters;
    float expected_effectiveness;  // 0.0 to 1.0
    std::vector<std::string> potential_side_effects;
};

/**
 * @brief Threat event (enriched security event)
 */
struct ThreatEvent {
    SecurityEvent base_event;
    float confidence;  // 0.0 to 1.0
    std::vector<FeatureAttribution> attributions;
    RecommendedAction recommended_action;
    std::string rationale;
    bool requires_alpha_permission;  // Always true in DIREWOLF
    UrgencyLevel urgency;
};

/**
 * @brief System state for context
 */
struct SystemState {
    size_t threats_today;
    size_t active_alerts;
    std::string health_status;  // "HEALTHY", "DEGRADED", "CRITICAL"
    float drl_confidence;
    std::vector<SecurityEvent> recent_events;
    std::map<std::string, std::string> component_status;
};

/**
 * @brief Conversation exchange
 */
struct ConversationExchange {
    std::chrono::system_clock::time_point timestamp;
    std::string user_input;
    std::string wolf_response;
    std::string context;  // What was happening at the time
};

/**
 * @brief User expertise level
 */
enum class ExpertiseLevel {
    NOVICE,
    INTERMEDIATE,
    EXPERT
};

/**
 * @brief User profile
 */
struct UserProfile {
    std::string user_id;
    std::string display_name;  // "Alpha"
    ExpertiseLevel technical_expertise;
    bool prefers_brief_responses;
    std::map<std::string, std::string> preferences;
    std::vector<std::string> decision_patterns;
};

/**
 * @brief Attack chain node
 */
struct AttackChainNode {
    SecurityEvent event;
    std::vector<std::string> parent_event_ids;
    std::vector<std::string> child_event_ids;
    std::string attack_technique;  // MITRE ATT&CK technique
    float confidence;
};

/**
 * @brief Complete attack chain
 */
struct AttackChain {
    std::string chain_id;
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;
    std::vector<AttackChainNode> nodes;
    std::string attack_type;
    Severity overall_severity;
    bool was_blocked;
};

/**
 * @brief Incident timeline
 */
struct IncidentTimeline {
    std::string incident_id;
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;
    std::vector<SecurityEvent> events;
    AttackChain attack_chain;
    std::vector<RecommendedAction> actions_taken;
    std::string outcome;
    std::map<std::string, std::string> metadata;
};

/**
 * @brief Helper functions
 */
inline std::string severityToString(Severity severity) {
    switch (severity) {
        case Severity::LOW: return "LOW";
        case Severity::MEDIUM: return "MEDIUM";
        case Severity::HIGH: return "HIGH";
        case Severity::CRITICAL: return "CRITICAL";
        case Severity::EMERGENCY: return "EMERGENCY";
        default: return "UNKNOWN";
    }
}

inline std::string urgencyToString(UrgencyLevel urgency) {
    switch (urgency) {
        case UrgencyLevel::ROUTINE: return "ROUTINE";
        case UrgencyLevel::ELEVATED: return "ELEVATED";
        case UrgencyLevel::CRITICAL: return "CRITICAL";
        case UrgencyLevel::EMERGENCY: return "EMERGENCY";
        default: return "UNKNOWN";
    }
}

inline UrgencyLevel severityToUrgency(Severity severity) {
    switch (severity) {
        case Severity::LOW:
        case Severity::MEDIUM:
            return UrgencyLevel::ROUTINE;
        case Severity::HIGH:
            return UrgencyLevel::ELEVATED;
        case Severity::CRITICAL:
            return UrgencyLevel::CRITICAL;
        case Severity::EMERGENCY:
            return UrgencyLevel::EMERGENCY;
        default:
            return UrgencyLevel::ROUTINE;
    }
}

} // namespace XAI
} // namespace DRLHSS
