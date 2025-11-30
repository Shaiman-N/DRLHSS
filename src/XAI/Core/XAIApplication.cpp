#include "XAI/Core/XAIApplication.hpp"
#include "XAI/Core/PluginManager.hpp"
#include "XAI/Core/EventBus.hpp"
#include "XAI/Core/ConfigManager.hpp"
#include "XAI/Core/Logger.hpp"
#include "XAI/Voice/VoiceInterface.hpp"

#include <QStandardPaths>
#include <QDir>
#include <iostream>

namespace DIREWOLF {
namespace XAI {

struct XAIApplication::Impl {
    std::unique_ptr<Logger> logger;
    std::unique_ptr<ConfigManager> configManager;
    std::unique_ptr<EventBus> eventBus;
    std::unique_ptr<PluginManager> pluginManager;
    std::unique_ptr<VoiceInterface> voiceInterface;
    bool initialized = false;
};

XAIApplication::XAIApplication(int& argc, char** argv)
    : QApplication(argc, argv)
    , impl_(std::make_unique<Impl>())
{
    setApplicationName("DIREWOLF XAI");
    setApplicationVersion("1.0.0");
    setOrganizationName("DIREWOLF");
    setOrganizationDomain("direwolf.security");
}

XAIApplication::~XAIApplication() {
    shutdown();
}

bool XAIApplication::initialize() {
    if (impl_->initialized) {
        return true;
    }

    std::cout << "Initializing DIREWOLF XAI System..." << std::endl;

    // Initialize in dependency order
    if (!initializeLogging()) {
        std::cerr << "Failed to initialize logging system" << std::endl;
        return false;
    }

    if (!initializeConfig()) {
        LOG_ERROR(impl_->logger.get(), "Failed to initialize configuration system");
        return false;
    }

    if (!initializeEventBus()) {
        LOG_ERROR(impl_->logger.get(), "Failed to initialize event bus");
        return false;
    }

    if (!initializePluginSystem()) {
        LOG_ERROR(impl_->logger.get(), "Failed to initialize plugin system");
        return false;
    }

    if (!initializeVoiceInterface()) {
        LOG_ERROR(impl_->logger.get(), "Failed to initialize voice interface");
        return false;
    }

    impl_->initialized = true;
    LOG_INFO(impl_->logger.get(), "DIREWOLF XAI System initialized successfully");
    
    emit initialized();
    return true;
}

void XAIApplication::shutdown() {
    if (!impl_->initialized) {
        return;
    }

    LOG_INFO(impl_->logger.get(), "Shutting down DIREWOLF XAI System...");

    // Shutdown in reverse order
    if (impl_->voiceInterface) {
        impl_->voiceInterface->shutdown();
        impl_->voiceInterface.reset();
    }

    if (impl_->pluginManager) {
        impl_->pluginManager->unloadAllPlugins();
        impl_->pluginManager.reset();
    }

    if (impl_->eventBus) {
        impl_->eventBus.reset();
    }

    if (impl_->configManager) {
        // Save configuration before shutdown
        auto configPath = QStandardPaths::writableLocation(QStandardPaths::AppConfigLocation);
        impl_->configManager->saveConfig(configPath.toStdString() + "/config.json");
        impl_->configManager.reset();
    }

    if (impl_->logger) {
        LOG_INFO(impl_->logger.get(), "DIREWOLF XAI System shutdown complete");
        impl_->logger->shutdown();
        impl_->logger.reset();
    }

    impl_->initialized = false;
    emit shutdownRequested();
}

bool XAIApplication::initializeLogging() {
    impl_->logger = std::make_unique<Logger>();

    // Create log directory
    auto logPath = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir().mkpath(logPath);

    std::filesystem::path logFile = logPath.toStdString() + "/direwolf_xai.log";
    
    if (!impl_->logger->initialize(logFile, LogLevel::Info)) {
        std::cerr << "Failed to initialize logger" << std::endl;
        return false;
    }

    impl_->logger->enableConsoleOutput(true);
    impl_->logger->enableFileOutput(true);
    impl_->logger->setMaxFileSize(10 * 1024 * 1024); // 10MB
    impl_->logger->setMaxFiles(5);

    return true;
}

bool XAIApplication::initializeConfig() {
    impl_->configManager = std::make_unique<ConfigManager>();

    // Try to load existing config
    auto configPath = QStandardPaths::writableLocation(QStandardPaths::AppConfigLocation);
    QDir().mkpath(configPath);

    std::filesystem::path configFile = configPath.toStdString() + "/config.json";
    
    if (std::filesystem::exists(configFile)) {
        if (!impl_->configManager->loadConfig(configFile)) {
            LOG_WARNING(impl_->logger.get(), "Failed to load config, using defaults");
            impl_->configManager->loadDefaultConfig();
        }
    } else {
        LOG_INFO(impl_->logger.get(), "No existing config found, creating default");
        impl_->configManager->loadDefaultConfig();
        impl_->configManager->saveConfig(configFile);
    }

    return true;
}

bool XAIApplication::initializeEventBus() {
    impl_->eventBus = std::make_unique<EventBus>();
    
    // Connect event bus signals to logger
    connect(impl_->eventBus.get(), &EventBus::eventError,
            [this](const QString& type, const QString& error) {
        LOG_ERROR(impl_->logger.get(), 
                 "Event error [" + type.toStdString() + "]: " + error.toStdString());
    });

    LOG_INFO(impl_->logger.get(), "Event bus initialized");
    return true;
}

bool XAIApplication::initializePluginSystem() {
    impl_->pluginManager = std::make_unique<PluginManager>();

    // Connect plugin manager signals
    connect(impl_->pluginManager.get(), &PluginManager::pluginLoaded,
            [this](const QString& name) {
        LOG_INFO(impl_->logger.get(), "Plugin loaded: " + name.toStdString());
    });

    connect(impl_->pluginManager.get(), &PluginManager::pluginUnloaded,
            [this](const QString& name) {
        LOG_INFO(impl_->logger.get(), "Plugin unloaded: " + name.toStdString());
    });

    connect(impl_->pluginManager.get(), &PluginManager::pluginError,
            [this](const QString& name, const QString& error) {
        LOG_ERROR(impl_->logger.get(), 
                 "Plugin error [" + name.toStdString() + "]: " + error.toStdString());
    });

    // Discover and load plugins
    auto pluginPath = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    pluginPath += "/plugins";
    QDir().mkpath(pluginPath);

    auto plugins = impl_->pluginManager->discoverPlugins(pluginPath.toStdString());
    LOG_INFO(impl_->logger.get(), 
             "Discovered " + std::to_string(plugins.size()) + " plugins");

    for (const auto& plugin : plugins) {
        impl_->pluginManager->loadPlugin(plugin);
    }

    return true;
}

PluginManager* XAIApplication::pluginManager() const {
    return impl_->pluginManager.get();
}

EventBus* XAIApplication::eventBus() const {
    return impl_->eventBus.get();
}

ConfigManager* XAIApplication::configManager() const {
    return impl_->configManager.get();
}

Logger* XAIApplication::logger() const {
    return impl_->logger.get();
}

VoiceInterface* XAIApplication::voiceInterface() const {
    return impl_->voiceInterface.get();
}

bool XAIApplication::initializeVoiceInterface() {
    impl_->voiceInterface = std::make_unique<VoiceInterface>(
        impl_->eventBus.get(),
        impl_->logger.get()
    );

    if (!impl_->voiceInterface->initialize()) {
        LOG_ERROR(impl_->logger.get(), "Failed to initialize voice interface");
        return false;
    }

    // Connect voice signals
    connect(impl_->voiceInterface.get(), &VoiceInterface::speechRecognized,
            [this](const QString& text, float confidence) {
        LOG_INFO(impl_->logger.get(), 
                "Speech recognized: " + text.toStdString() + 
                " (confidence: " + std::to_string(confidence) + ")");
    });

    LOG_INFO(impl_->logger.get(), "Voice interface initialized");
    return true;
}

} // namespace XAI
} // namespace DIREWOLF
