#pragma once

#include <memory>
#include <string>
#include <filesystem>
#include <optional>
#include <QObject>
#include <QVariant>

namespace DIREWOLF {
namespace XAI {

/**
 * @brief Configuration management system
 * 
 * Handles loading, saving, and accessing application configuration.
 * Supports JSON format with hierarchical keys and type-safe access.
 */
class ConfigManager : public QObject {
    Q_OBJECT

public:
    explicit ConfigManager(QObject* parent = nullptr);
    ~ConfigManager() override;

    // Configuration file operations
    bool loadConfig(const std::filesystem::path& path);
    bool saveConfig(const std::filesystem::path& path);
    bool loadDefaultConfig();

    // Value access (type-safe)
    template<typename T>
    T getValue(const std::string& key, const T& defaultValue = T{}) const;

    template<typename T>
    void setValue(const std::string& key, const T& value);

    // Generic access
    QVariant getVariant(const std::string& key, 
                       const QVariant& defaultValue = QVariant()) const;
    void setVariant(const std::string& key, const QVariant& value);

    // Key operations
    bool hasKey(const std::string& key) const;
    void removeKey(const std::string& key);
    std::vector<std::string> getAllKeys() const;
    std::vector<std::string> getKeysInSection(const std::string& section) const;

    // Configuration sections
    void beginSection(const std::string& section);
    void endSection();
    std::string currentSection() const;

    // Validation
    bool validate() const;
    std::vector<std::string> getValidationErrors() const;

signals:
    void configLoaded(const QString& path);
    void configSaved(const QString& path);
    void valueChanged(const QString& key, const QVariant& value);
    void configError(const QString& error);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Template implementations
template<typename T>
T ConfigManager::getValue(const std::string& key, const T& defaultValue) const {
    QVariant variant = getVariant(key);
    if (!variant.isValid() || !variant.canConvert<T>()) {
        return defaultValue;
    }
    return variant.value<T>();
}

template<typename T>
void ConfigManager::setValue(const std::string& key, const T& value) {
    setVariant(key, QVariant::fromValue(value));
}

} // namespace XAI
} // namespace DIREWOLF
