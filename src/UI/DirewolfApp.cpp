/**
 * @file DirewolfApp.cpp
 * @brief DIREWOLF Main Application
 * 
 * Qt-based desktop application with system tray, dashboard, and chat interface.
 */

#include "UI/DirewolfApp.hpp"
#include <QApplication>
#include <QSystemTrayIcon>
#include <QMenu>
#include <QAction>
#include <QMessageBox>
#include <QTimer>
#include <iostream>

namespace ui {

DirewolfApp::DirewolfApp(int& argc, char** argv)
    : app_(std::make_unique<QApplication>(argc, argv)),
      current_status_(SystemStatus::IDLE) {
    
    // Set application metadata
    QApplication::setApplicationName("DIREWOLF");
    QApplication::setApplicationVersion("1.0.0");
    QApplication::setOrganizationName("DRLHSS");
    
    std::cout << "[DIREWOLF] Application initialized" << std::endl;
}

DirewolfApp::~DirewolfApp() {
    if (tray_icon_) {
        tray_icon_->hide();
    }
}

bool DirewolfApp::initialize(const std::string& db_path, const std::string& model_path) {
    try {
        // Initialize DRLHSS bridge
        bridge_ = std::make_unique<xai::DRLHSSBridge>(db_path, model_path);
        
        if (!bridge_->initialize()) {
            std::cerr << "[DIREWOLF] Failed to initialize DRLHSS bridge" << std::endl;
            return false;
        }
        
        // Create system tray icon
        createSystemTray();
        
        // Create main windows (hidden initially)
        createDashboard();
        createChatWindow();
        createPermissionDialog();
        
        // Setup update timer
        update_timer_ = std::make_unique<QTimer>();
        QObject::connect(update_timer_.get(), &QTimer::timeout, [this]() {
            updateSystemStatus();
        });
        update_timer_->start(1000); // Update every second
        
        // Show system tray
        if (tray_icon_) {
            tray_icon_->show();
            tray_icon_->showMessage(
                "DIREWOLF Active",
                "Your security guardian is watching.",
                QSystemTrayIcon::Information,
                3000
            );
        }
        
        std::cout << "[DIREWOLF] Initialization complete" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[DIREWOLF] Initialization error: " << e.what() << std::endl;
        return false;
    }
}

int DirewolfApp::run() {
    if (!app_) {
        return -1;
    }
    
    return app_->exec();
}

void DirewolfApp::shutdown() {
    std::cout << "[DIREWOLF] Shutting down..." << std::endl;
    
    // Stop update timer
    if (update_timer_) {
        update_timer_->stop();
    }
    
    // Hide windows
    if (dashboard_) {
        dashboard_->hide();
    }
    
    if (chat_window_) {
        chat_window_->hide();
    }
    
    // Hide tray icon
    if (tray_icon_) {
        tray_icon_->hide();
    }
    
    // Quit application
    if (app_) {
        app_->quit();
    }
}

// ========== System Tray ==========

void DirewolfApp::createSystemTray() {
    // Create tray icon
    tray_icon_ = std::make_unique<QSystemTrayIcon>();
    
    // Set initial icon (would load from resources in production)
    updateTrayIcon(SystemStatus::IDLE);
    
    // Create context menu
    tray_menu_ = std::make_unique<QMenu>();
    
    // Add menu actions
    auto* show_dashboard = tray_menu_->addAction("Show Dashboard");
    QObject::connect(show_dashboard, &QAction::triggered, [this]() {
        showDashboard();
    });
    
    auto* show_chat = tray_menu_->addAction("Chat with Wolf");
    QObject::connect(show_chat, &QAction::triggered, [this]() {
        showChatWindow();
    });
    
    tray_menu_->addSeparator();
    
    // Status submenu
    auto* status_menu = tray_menu_->addMenu("System Status");
    status_action_ = status_menu->addAction("Status: Idle");
    status_action_->setEnabled(false);
    
    tray_menu_->addSeparator();
    
    auto* quit_action = tray_menu_->addAction("Quit DIREWOLF");
    QObject::connect(quit_action, &QAction::triggered, [this]() {
        shutdown();
    });
    
    // Set menu
    tray_icon_->setContextMenu(tray_menu_.get());
    
    // Connect activation signal
    QObject::connect(tray_icon_.get(), &QSystemTrayIcon::activated,
        [this](QSystemTrayIcon::ActivationReason reason) {
            if (reason == QSystemTrayIcon::DoubleClick) {
                showDashboard();
            }
        });
    
    std::cout << "[DIREWOLF] System tray created" << std::endl;
}

void DirewolfApp::updateTrayIcon(SystemStatus status) {
    if (!tray_icon_) {
        return;
    }
    
    current_status_ = status;
    
    // In production, would load actual icon files
    QString tooltip;
    
    switch (status) {
        case SystemStatus::IDLE:
            tooltip = "DIREWOLF - Idle";
            break;
        case SystemStatus::MONITORING:
            tooltip = "DIREWOLF - Monitoring";
            break;
        case SystemStatus::ALERT:
            tooltip = "DIREWOLF - Alert!";
            break;
        case SystemStatus::CRITICAL:
            tooltip = "DIREWOLF - CRITICAL THREAT!";
            break;
    }
    
    tray_icon_->setToolTip(tooltip);
    
    // Update status action
    if (status_action_) {
        status_action_->setText("Status: " + tooltip);
    }
}

void DirewolfApp::showNotification(
    const std::string& title,
    const std::string& message,
    NotificationLevel level
) {
    if (!tray_icon_) {
        return;
    }
    
    QSystemTrayIcon::MessageIcon icon;
    
    switch (level) {
        case NotificationLevel::INFO:
            icon = QSystemTrayIcon::Information;
            break;
        case NotificationLevel::WARNING:
            icon = QSystemTrayIcon::Warning;
            break;
        case NotificationLevel::CRITICAL:
            icon = QSystemTrayIcon::Critical;
            break;
    }
    
    tray_icon_->showMessage(
        QString::fromStdString(title),
        QString::fromStdString(message),
        icon,
        5000 // 5 seconds
    );
}

// ========== Dashboard ==========

void DirewolfApp::createDashboard() {
    dashboard_ = std::make_unique<QWidget>();
    dashboard_->setWindowTitle("DIREWOLF Dashboard");
    dashboard_->resize(800, 600);
    
    // In production, would load QML or create complex layout
    // For now, create basic structure
    
    std::cout << "[DIREWOLF] Dashboard created" << std::endl;
}

void DirewolfApp::showDashboard() {
    if (dashboard_) {
        dashboard_->show();
        dashboard_->raise();
        dashboard_->activateWindow();
    }
}

void DirewolfApp::hideDashboard() {
    if (dashboard_) {
        dashboard_->hide();
    }
}

// ========== Chat Window ==========

void DirewolfApp::createChatWindow() {
    chat_window_ = std::make_unique<QWidget>();
    chat_window_->setWindowTitle("Chat with Wolf");
    chat_window_->resize(600, 400);
    
    std::cout << "[DIREWOLF] Chat window created" << std::endl;
}

void DirewolfApp::showChatWindow() {
    if (chat_window_) {
        chat_window_->show();
        chat_window_->raise();
        chat_window_->activateWindow();
    }
}

void DirewolfApp::hideChatWindow() {
    if (chat_window_) {
        chat_window_->hide();
    }
}

// ========== Permission Dialog ==========

void DirewolfApp::createPermissionDialog() {
    permission_dialog_ = std::make_unique<QDialog>();
    permission_dialog_->setWindowTitle("DIREWOLF - Permission Required");
    permission_dialog_->setModal(true);
    permission_dialog_->resize(500, 400);
    
    std::cout << "[DIREWOLF] Permission dialog created" << std::endl;
}

void DirewolfApp::showPermissionRequest(const xai::PermissionRequest& request) {
    if (!permission_dialog_) {
        return;
    }
    
    // In production, would populate dialog with request details
    // and connect approve/reject buttons
    
    permission_dialog_->show();
    permission_dialog_->raise();
    permission_dialog_->activateWindow();
    
    // Show notification
    showNotification(
        "Permission Required",
        "Wolf needs your decision on a security action.",
        NotificationLevel::WARNING
    );
}

// ========== Update Loop ==========

void DirewolfApp::updateSystemStatus() {
    if (!bridge_) {
        return;
    }
    
    try {
        // Get system snapshot
        auto snapshot_dict = bridge_->getSystemSnapshot();
        
        // Update status based on active alerts
        // (In production, would parse snapshot_dict)
        
        // For now, cycle through statuses for demonstration
        static int counter = 0;
        counter++;
        
        if (counter % 60 == 0) {
            // Every 60 seconds, show monitoring status
            updateTrayIcon(SystemStatus::MONITORING);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[DIREWOLF] Update error: " << e.what() << std::endl;
    }
}

} // namespace ui


// ========== Main Entry Point ==========

int main(int argc, char* argv[]) {
    try {
        // Create application
        ui::DirewolfApp app(argc, argv);
        
        // Initialize with database and model paths
        if (!app.initialize("drlhss.db", "models/drl_model.onnx")) {
            std::cerr << "Failed to initialize DIREWOLF" << std::endl;
            return 1;
        }
        
        std::cout << "DIREWOLF is running. Check system tray." << std::endl;
        
        // Run application
        return app.run();
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
