#include "Telemetry/TelemetryEvent.hpp"
#include <sstream>
#include <iomanip>

namespace telemetry {

std::string eventTypeToString(EventType type) {
    switch (type) {
        case EventType::PROCESS: return "process";
        case EventType::FILE: return "file";
        case EventType::REGISTRY: return "registry";
        case EventType::NETWORK: return "network";
        case EventType::SYSCALL: return "syscall";
        case EventType::API_CALL: return "api_call";
        case EventType::MEMORY: return "memory";
        case EventType::IPC: return "ipc";
        case EventType::SANDBOX: return "sandbox";
        case EventType::STATIC_ANALYSIS: return "static_analysis";
        case EventType::USER_BEHAVIOR: return "user_behavior";
        default: return "unknown";
    }
}

EventType stringToEventType(const std::string& str) {
    if (str == "process") return EventType::PROCESS;
    if (str == "file") return EventType::FILE;
    if (str == "registry") return EventType::REGISTRY;
    if (str == "network") return EventType::NETWORK;
    if (str == "syscall") return EventType::SYSCALL;
    if (str == "api_call") return EventType::API_CALL;
    if (str == "memory") return EventType::MEMORY;
    if (str == "ipc") return EventType::IPC;
    if (str == "sandbox") return EventType::SANDBOX;
    if (str == "static_analysis") return EventType::STATIC_ANALYSIS;
    if (str == "user_behavior") return EventType::USER_BEHAVIOR;
    return EventType::UNKNOWN;
}

TelemetryEvent::TelemetryEvent() 
    : type(EventType::UNKNOWN), 
      timestamp(std::chrono::system_clock::now()),
      pid(0),
      threat_score(0.0f),
      is_suspicious(false) {}

TelemetryEvent::TelemetryEvent(EventType t, int process_id, const std::string& proc_name)
    : type(t),
      timestamp(std::chrono::system_clock::now()),
      pid(process_id),
      process_name(proc_name),
      threat_score(0.0f),
      is_suspicious(false) {}

std::string TelemetryEvent::toJSON() const {
    std::ostringstream oss;
    oss << "{";
    oss << "\"type\":\"" << eventTypeToString(type) << "\",";
    
    auto time_t_val = std::chrono::system_clock::to_time_t(timestamp);
    oss << "\"timestamp\":" << time_t_val << ",";
    
    oss << "\"pid\":" << pid << ",";
    oss << "\"process\":\"" << process_name << "\",";
    oss << "\"process_path\":\"" << process_path << "\",";
    oss << "\"threat_score\":" << threat_score << ",";
    oss << "\"is_suspicious\":" << (is_suspicious ? "true" : "false");
    
    if (!attributes.empty()) {
        oss << ",\"attributes\":{";
        bool first = true;
        for (const auto& [key, value] : attributes) {
            if (!first) oss << ",";
            oss << "\"" << key << "\":\"" << value << "\"";
            first = false;
        }
        oss << "}";
    }
    
    oss << "}";
    return oss.str();
}

void TelemetryEvent::setAttribute(const std::string& key, const std::string& value) {
    attributes[key] = value;
}

std::string TelemetryEvent::getAttribute(const std::string& key, const std::string& default_val) const {
    auto it = attributes.find(key);
    return (it != attributes.end()) ? it->second : default_val;
}

bool TelemetryEvent::hasAttribute(const std::string& key) const {
    return attributes.find(key) != attributes.end();
}

TelemetryEvent TelemetryEvent::createProcessEvent(int pid, const std::string& name,
                                                  const std::string& path, const std::string& action) {
    TelemetryEvent event(EventType::PROCESS, pid, name);
    event.process_path = path;
    event.setAttribute("action", action);
    return event;
}

TelemetryEvent TelemetryEvent::createFileEvent(int pid, const std::string& proc_name,
                                               const std::string& file_path, const std::string& operation) {
    TelemetryEvent event(EventType::FILE, pid, proc_name);
    event.setAttribute("path", file_path);
    event.setAttribute("operation", operation);
    return event;
}

TelemetryEvent TelemetryEvent::createNetworkEvent(int pid, const std::string& proc_name,
                                                  const std::string& dst_ip, int dst_port,
                                                  const std::string& protocol) {
    TelemetryEvent event(EventType::NETWORK, pid, proc_name);
    event.setAttribute("dst_ip", dst_ip);
    event.setAttribute("dst_port", std::to_string(dst_port));
    event.setAttribute("protocol", protocol);
    event.setAttribute("direction", "outbound");
    return event;
}

TelemetryEvent TelemetryEvent::createSyscallEvent(int pid, const std::string& proc_name,
                                                  const std::string& syscall, const std::string& target) {
    TelemetryEvent event(EventType::SYSCALL, pid, proc_name);
    event.setAttribute("syscall", syscall);
    event.setAttribute("target", target);
    return event;
}

TelemetryEvent TelemetryEvent::createRegistryEvent(int pid, const std::string& proc_name,
                                                   const std::string& key, const std::string& operation) {
    TelemetryEvent event(EventType::REGISTRY, pid, proc_name);
    event.setAttribute("key", key);
    event.setAttribute("operation", operation);
    return event;
}

} // namespace telemetry
