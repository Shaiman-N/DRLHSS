#include "XAI/Core/PluginManager.hpp"

#include <map>
#include <mutex>
#include <QLibrary>
#include <QDir>

namespace DIREWOLF {
namespace XAI {

struct PluginManager::Impl {
    std::map<std::string, std::unique_ptr<IPlugin>> plugins;
    std::map<std::string, PluginInfo> pluginInfo;
    std::map<std::string, std::unique_ptr<QLibrary>> libraries;
    mutable std::mutex mutex;
};

PluginManager::PluginManager(QObject* parent)
    : QObject(parent)
    , impl_(std::make_unique<Impl>())
{
}

PluginManager::~PluginManager() {
    unloadAllPlugins();
}

bool PluginManager::loadPlugin(const std::filesystem::path& path) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    if (!std::filesystem::exists(path)) {
        emit pluginError("", QString("Plugin file not found: %1")
                        .arg(QString::fromStdString(path.string())));
        return false;
    }

    // Load library
    auto library = std::make_unique<QLibrary>(QString::fromStdString(path.string()));
    
    if (!library->load()) {
        emit pluginError("", QString("Failed to load plugin library: %1")
                        .arg(library->errorString()));
        return false;
    }

    // Get plugin factory function
    typedef IPlugin* (*CreatePluginFunc)();
    auto createPlugin = reinterpret_cast<CreatePluginFunc>(
        library->resolve("createPlugin"));

    if (!createPlugin) {
        emit pluginError("", "Plugin does not export createPlugin function");
        library->unload();
        return false;
    }

    // Create plugin instance
    IPlugin* plugin = createPlugin();
    if (!plugin) {
        emit pluginError("", "Failed to create plugin instance");
        library->unload();
        return false;
    }

    std::string pluginName = plugin->getName();

    // Check if already loaded
    if (impl_->plugins.find(pluginName) != impl_->plugins.end()) {
        emit pluginError(QString::fromStdString(pluginName), 
                        "Plugin already loaded");
        delete plugin;
        library->unload();
        return false;
    }

    // Initialize plugin
    if (!plugin->initialize()) {
        emit pluginError(QString::fromStdString(pluginName), 
                        "Plugin initialization failed");
        delete plugin;
        library->unload();
        return false;
    }

    // Store plugin info
    PluginInfo info;
    info.name = pluginName;
    info.version = plugin->getVersion();
    info.description = plugin->getDescription();
    info.path = path;
    info.loaded = true;

    impl_->plugins[pluginName] = std::unique_ptr<IPlugin>(plugin);
    impl_->pluginInfo[pluginName] = info;
    impl_->libraries[pluginName] = std::move(library);

    emit pluginLoaded(QString::fromStdString(pluginName));
    return true;
}

bool PluginManager::unloadPlugin(const std::string& name) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    auto it = impl_->plugins.find(name);
    if (it == impl_->plugins.end()) {
        return false;
    }

    // Shutdown plugin
    it->second->shutdown();
    impl_->plugins.erase(it);

    // Unload library
    auto libIt = impl_->libraries.find(name);
    if (libIt != impl_->libraries.end()) {
        libIt->second->unload();
        impl_->libraries.erase(libIt);
    }

    // Remove info
    impl_->pluginInfo.erase(name);

    emit pluginUnloaded(QString::fromStdString(name));
    return true;
}

void PluginManager::unloadAllPlugins() {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    // Shutdown all plugins
    for (auto& [name, plugin] : impl_->plugins) {
        plugin->shutdown();
        emit pluginUnloaded(QString::fromStdString(name));
    }

    impl_->plugins.clear();
    
    // Unload all libraries
    for (auto& [name, library] : impl_->libraries) {
        library->unload();
    }
    impl_->libraries.clear();
    
    impl_->pluginInfo.clear();
}

std::vector<IPlugin*> PluginManager::getLoadedPlugins() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    std::vector<IPlugin*> result;
    result.reserve(impl_->plugins.size());
    
    for (const auto& [name, plugin] : impl_->plugins) {
        result.push_back(plugin.get());
    }
    
    return result;
}

IPlugin* PluginManager::getPlugin(const std::string& name) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    auto it = impl_->plugins.find(name);
    if (it != impl_->plugins.end()) {
        return it->second.get();
    }
    
    return nullptr;
}

std::vector<PluginInfo> PluginManager::getPluginInfo() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    std::vector<PluginInfo> result;
    result.reserve(impl_->pluginInfo.size());
    
    for (const auto& [name, info] : impl_->pluginInfo) {
        result.push_back(info);
    }
    
    return result;
}

bool PluginManager::isPluginLoaded(const std::string& name) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->plugins.find(name) != impl_->plugins.end();
}

std::vector<std::filesystem::path> PluginManager::discoverPlugins(
    const std::filesystem::path& directory) const 
{
    std::vector<std::filesystem::path> plugins;

    if (!std::filesystem::exists(directory)) {
        return plugins;
    }

    QDir dir(QString::fromStdString(directory.string()));
    QStringList filters;
    
#ifdef _WIN32
    filters << "*.dll";
#elif defined(__APPLE__)
    filters << "*.dylib";
#else
    filters << "*.so";
#endif

    dir.setNameFilters(filters);
    QFileInfoList files = dir.entryInfoList(QDir::Files);

    for (const auto& fileInfo : files) {
        plugins.push_back(fileInfo.absoluteFilePath().toStdString());
    }

    return plugins;
}

} // namespace XAI
} // namespace DIREWOLF
