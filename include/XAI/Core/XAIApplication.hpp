#pragma once

#include <memory>
#include <string>
#include <QApplication>

namespace DIREWOLF {
namespace XAI {

class PluginManager;
class EventBus;
class ConfigManager;
class Logger;
class VoiceInterface;

/**
 * @brief Main XAI Application class
 * 
 * Core application that manages all XAI subsystems including
 * plugin loading, event handling, and configuration.
 */
class XAIApplication : public QApplication {
    Q_OBJECT

public:
    explicit XAIApplication(int& argc, char** argv);
    ~XAIApplication() override;

    // Initialization and shutdown
    bool initialize();
    void shutdown();

    // Component access
    PluginManager* pluginManager() const;
    EventBus* eventBus() const;
    ConfigManager* configManager() const;
    Logger* logger() const;
    VoiceInterface* voiceInterface() const;

    // Application info
    std::string version() const { return "1.0.0"; }
    std::string name() const { return "DIREWOLF XAI"; }

signals:
    void initialized();
    void shutdownRequested();
    void errorOccurred(const QString& error);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    bool initializeLogging();
    bool initializeConfig();
    bool initializeEventBus();
    bool initializePluginSystem();
    bool initializeVoiceInterface();
};

} // namespace XAI
} // namespace DIREWOLF
