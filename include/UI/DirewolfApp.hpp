/**
 * @file DirewolfApp.hpp
 * @brief DIREWOLF Main Application
 * 
 * Qt-based desktop application with system tray, dashboard, and chat interface.
 */

#pragma once

#include "XAI/DRLHSSBridge.hpp"
#include "XAI/XAITypes.hpp"
#include <QApplication>
#include <QSystemTrayIcon>
#include <QMenu>
#include <QAction>
#include <QWidget>
#include <QDialog>
#include <QTimer>
#include <memory>
#include <string>

namespace ui {

/**
 * @brief System status levels
 */
enum class SystemStatus {
    IDLE,           // No threats, system idle
    MONITORING,     // Actively monitoring
    ALERT,          // Threat detected, awaiting decision
    CRITICAL        // Critical threat, immediate attention needed
};

/**
 * @brief Notification levels
 */
enum class NotificationLevel {
    INFO,
    WARNING,
    CRITICAL
};

/**
 * @brief DIREWOLF Main Application
 * 
 * Manages system tray, dashboard, chat interface, and permission dialogs.
 */
class DirewolfApp {
public:
    /**
     * @brief Constructor
     * @param argc Argument count
     * @param argv Argument values
     */
    DirewolfApp(int& argc, char** argv);
    
    /**
     * @brief Destructor
     */
    ~DirewolfApp();
    
    /**
     * @brief Initialize application
     * @param db_path Path to DRLHSS database
     * @param model_path Path to DRL model
     * @return True if successful
     */
    bool initialize(const std::string& db_path, const std::string& model_path);
    
    /**
     * @brief Run application event loop
     * @return Exit code
     */
    int run();
    
    /**
     * @brief Shutdown application
     */
    void shutdown();
    
    // ========== System Tray ==========
    
    /**
     * @brief Update tray icon based on system status
     * @param status Current system status
     */
    void updateTrayIcon(SystemStatus status);
    
    /**
     * @brief Show notification
     * @param title Notification title
     * @param message Notification message
     * @param level Notification level
     */
    void showNotification(
        const std::string& title,
        const std::string& message,
        NotificationLevel level = NotificationLevel::INFO
    );
    
    // ========== Dashboard ==========
    
    /**
     * @brief Show dashboard window
     */
    void showDashboard();
    
    /**
     * @brief Hide dashboard window
     */
    void hideDashboard();
    
    // ========== Chat Window ==========
    
    /**
     * @brief Show chat window
     */
    void showChatWindow();
    
    /**
     * @brief Hide chat window
     */
    void hideChatWindow();
    
    // ========== Permission Dialog ==========
    
    /**
     * @brief Show permission request dialog
     * @param request Permission request
     */
    void showPermissionRequest(const xai::PermissionRequest& request);

private:
    // Qt application
    std::unique_ptr<QApplication> app_;
    
    // DRLHSS bridge
    std::unique_ptr<xai::DRLHSSBridge> bridge_;
    
    // System tray
    std::unique_ptr<QSystemTrayIcon> tray_icon_;
    std::unique_ptr<QMenu> tray_menu_;
    QAction* status_action_ = nullptr;
    
    // Windows
    std::unique_ptr<QWidget> dashboard_;
    std::unique_ptr<QWidget> chat_window_;
    std::unique_ptr<QDialog> permission_dialog_;
    
    // Update timer
    std::unique_ptr<QTimer> update_timer_;
    
    // State
    SystemStatus current_status_;
    
    // Private methods
    void createSystemTray();
    void createDashboard();
    void createChatWindow();
    void createPermissionDialog();
    void updateSystemStatus();
};

} // namespace ui
