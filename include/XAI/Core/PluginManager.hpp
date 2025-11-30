#pragma once

#include <memory>
#include <string>
#include <vector>
#include <filesystem>
#include <QObject>

namespace DIREWOLF {
namespace XAI {

/**
 * @brief Plugin interface that all plugins must implement
 */
class IPlugin {
public:
    virtual ~IPlugin() = default;

    virtual bool initialize() = 0;
    virtual void shutdown() = 0;
    virtual std::string getName() const = 0;
    virtual std::string getVersion() const = 0;
    virtual std::string getDescription() const = 0;
};

/**
 * @brief Plugin metadata
 */
struct PluginInfo {
    std::string name;
    std::string version;
    std::string description;
    std::filesystem::path path;
    bool loaded = false;
};

/**
 * @brief Manages dynamic plugin loading and lifecycle
 */
class PluginManager : public QObject {
    Q_OBJECT

public:
    explicit PluginManager(QObject* parent = nullptr);
    ~PluginManager() override;

    // Plugin loading
    bool loadPlugin(const std::filesystem::path& path);
    bool unloadPlugin(const std::string& name);
    void unloadAllPlugins();

    // Plugin queries
    std::vector<IPlugin*> getLoadedPlugins() const;
    IPlugin* getPlugin(const std::string& name) const;
    std::vector<PluginInfo> getPluginInfo() const;
    bool isPluginLoaded(const std::string& name) const;

    // Plugin discovery
    std::vector<std::filesystem::path> discoverPlugins(
        const std::filesystem::path& directory) const;

signals:
    void pluginLoaded(const QString& name);
    void pluginUnloaded(const QString& name);
    void pluginError(const QString& name, const QString& error);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace XAI
} // namespace DIREWOLF
