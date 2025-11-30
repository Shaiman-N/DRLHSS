/**
 * @file DRLHSSBridge.hpp
 * @brief Python-C++ Bridge for DIREWOLF
 * 
 * Provides Python bindings for all DRLHSS C++ components using pybind11.
 * Enables Wolf's Python AI to interact with C++ security systems.
 */

#pragma once

#include "XAI/XAITypes.hpp"
#include "XAI/PermissionRequestManager.hpp"
#include "XAI/XAIDataAggregator.hpp"
#include "XAI/ActionExecutor.hpp"
#include "Telemetry/TelemetryEvent.hpp"
#include "DRL/DRLOrchestrator.hpp"
#include "DB/DatabaseManager.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <memory>
#include <string>

namespace py = pybind11;

namespace xai {

/**
 * @brief DRLHSS Bridge - Python-C++ Integration
 * 
 * Unified interface for Python code to access all DRLHSS components.
 * Simplifies integration and provides high-level API.
 */
class DRLHSSBridge {
public:
    /**
     * @brief Constructor
     * @param db_path Path to DRLHSS database
     * @param model_path Path to DRL model
     */
    DRLHSSBridge(const std::string& db_path, const std::string& model_path);
    
    /**
     * @brief Destructor
     */
    ~DRLHSSBridge();
    
    /**
     * @brief Initialize all components
     * @return True if successful
     */
    bool initialize();
    
    // ========== Permission System ==========
    
    /**
     * @brief Submit permission request
     * @param threat_info Threat information
     * @param recommended_action Recommended action
     * @param urgency Urgency level
     * @param confidence Confidence score
     * @return Request ID
     */
    std::string submitPermissionRequest(
        const py::dict& threat_info,
        const std::string& recommended_action,
        const std::string& urgency,
        float confidence
    );
    
    /**
     * @brief Wait for Alpha's decision
     * @param request_id Request ID
     * @param timeout_ms Timeout in milliseconds
     * @return Decision dict
     */
    py::dict waitForDecision(const std::string& request_id, int timeout_ms);
    
    /**
     * @brief Provide Alpha's decision
     * @param request_id Request ID
     * @param decision Decision (approved/rejected)
     * @param alternative Alternative action (optional)
     */
    void provideDecision(
        const std::string& request_id,
        const std::string& decision,
        const std::string& alternative = ""
    );
    
    // ========== Data Aggregation ==========
    
    /**
     * @brief Get system snapshot
     * @return System snapshot dict
     */
    py::dict getSystemSnapshot();
    
    /**
     * @brief Get recent events
     * @param limit Maximum number of events
     * @return List of event dicts
     */
    py::list getRecentEvents(int limit = 100);
    
    /**
     * @brief Get recent threats
     * @param limit Maximum number of threats
     * @return List of threat dicts
     */
    py::list getRecentThreats(int limit = 50);
    
    /**
     * @brief Get component status
     * @param component Component name
     * @return Status string
     */
    std::string getComponentStatus(const std::string& component);
    
    /**
     * @brief Update component status
     * @param component Component name
     * @param status Status string
     */
    void updateComponentStatus(const std::string& component, const std::string& status);
    
    /**
     * @brief Get threat metrics
     * @return Metrics dict
     */
    py::dict getThreatMetrics();
    
    // ========== Action Execution ==========
    
    /**
     * @brief Execute action
     * @param action_type Action type
     * @param target Target (IP, file path, PID, etc.)
     * @param params Additional parameters
     * @return Action result dict
     */
    py::dict executeAction(
        const std::string& action_type,
        const std::string& target,
        const py::dict& params = py::dict()
    );
    
    /**
     * @brief Get action history
     * @param limit Maximum number of actions
     * @return List of action result dicts
     */
    py::list getActionHistory(int limit = 100);
    
    /**
     * @brief Rollback action
     * @param action_id Action ID
     * @return Action result dict
     */
    py::dict rollbackAction(const std::string& action_id);
    
    // ========== DRL Integration ==========
    
    /**
     * @brief Process telemetry with DRL
     * @param telemetry_dict Telemetry data dict
     * @return Detection response dict
     */
    py::dict processTelemetry(const py::dict& telemetry_dict);
    
    /**
     * @brief Get DRL confidence
     * @return Confidence score
     */
    float getDRLConfidence();
    
    /**
     * @brief Reload DRL model
     * @param model_path Path to new model
     * @return True if successful
     */
    bool reloadDRLModel(const std::string& model_path);
    
    // ========== Statistics ==========
    
    /**
     * @brief Get all system statistics
     * @return Statistics dict
     */
    py::dict getSystemStatistics();

private:
    // Components
    std::unique_ptr<PermissionRequestManager> permission_manager_;
    std::unique_ptr<XAIDataAggregator> data_aggregator_;
    std::unique_ptr<ActionExecutor> action_executor_;
    std::unique_ptr<drl::DRLOrchestrator> drl_orchestrator_;
    
    // Configuration
    std::string db_path_;
    std::string model_path_;
    
    // Helper methods
    ThreatInfo dictToThreatInfo(const py::dict& dict);
    py::dict threatInfoToDict(const ThreatInfo& threat);
    py::dict actionResultToDict(const ActionResult& result);
    py::dict systemSnapshotToDict(const SystemSnapshot& snapshot);
};

/**
 * @brief Initialize Python module
 * 
 * This function is called by pybind11 to create the Python module.
 */
PYBIND11_MODULE(drlhss_bridge, m) {
    m.doc() = "DIREWOLF DRLHSS Bridge - Python-C++ Integration";
    
    // DRLHSSBridge class
    py::class_<DRLHSSBridge>(m, "DRLHSSBridge")
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("db_path"),
             py::arg("model_path"))
        .def("initialize", &DRLHSSBridge::initialize)
        
        // Permission system
        .def("submit_permission_request", &DRLHSSBridge::submitPermissionRequest,
             py::arg("threat_info"),
             py::arg("recommended_action"),
             py::arg("urgency"),
             py::arg("confidence"))
        .def("wait_for_decision", &DRLHSSBridge::waitForDecision,
             py::arg("request_id"),
             py::arg("timeout_ms"))
        .def("provide_decision", &DRLHSSBridge::provideDecision,
             py::arg("request_id"),
             py::arg("decision"),
             py::arg("alternative") = "")
        
        // Data aggregation
        .def("get_system_snapshot", &DRLHSSBridge::getSystemSnapshot)
        .def("get_recent_events", &DRLHSSBridge::getRecentEvents,
             py::arg("limit") = 100)
        .def("get_recent_threats", &DRLHSSBridge::getRecentThreats,
             py::arg("limit") = 50)
        .def("get_component_status", &DRLHSSBridge::getComponentStatus,
             py::arg("component"))
        .def("update_component_status", &DRLHSSBridge::updateComponentStatus,
             py::arg("component"),
             py::arg("status"))
        .def("get_threat_metrics", &DRLHSSBridge::getThreatMetrics)
        
        // Action execution
        .def("execute_action", &DRLHSSBridge::executeAction,
             py::arg("action_type"),
             py::arg("target"),
             py::arg("params") = py::dict())
        .def("get_action_history", &DRLHSSBridge::getActionHistory,
             py::arg("limit") = 100)
        .def("rollback_action", &DRLHSSBridge::rollbackAction,
             py::arg("action_id"))
        
        // DRL integration
        .def("process_telemetry", &DRLHSSBridge::processTelemetry,
             py::arg("telemetry_dict"))
        .def("get_drl_confidence", &DRLHSSBridge::getDRLConfidence)
        .def("reload_drl_model", &DRLHSSBridge::reloadDRLModel,
             py::arg("model_path"))
        
        // Statistics
        .def("get_system_statistics", &DRLHSSBridge::getSystemStatistics);
}

} // namespace xai
