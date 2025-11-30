#include "XAI/Core/ConfigManager.hpp"

#include <fstream>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QFile>

namespace DIREWOLF {
namespace XAI {

struct ConfigManager::Impl {
    QJsonObject config;
    std::string currentSection;
    std::vector<std::string> validationErrors;
};

ConfigManager::ConfigManager(QObject* parent)
    : QObject(parent)
    , impl_(std::make_unique<Impl>())
{
}

ConfigManager::~ConfigManager() = default;

bool ConfigManager::loadConfig(const std::filesystem::path& path) {
    QFile file(QString::fromStdString(path.string()));
    
    if (!file.open(QIODevice::ReadOnly)) {
        emit configError(QString("Failed to open config file: %1")
                        .arg(QString::fromStdString(path.string())));
        return false;
    }

    QByteArray data = file.readAll();
    file.close();

    QJsonParseError error;
    QJsonDocument doc = QJsonDocument::fromJson(data, &error);

    if (error.error != QJsonParseError::NoError) {
        emit configError(QString("JSON parse error: %1").arg(error.errorString()));
        return false;
    }

    if (!doc.isObject()) {
        emit configError("Config file must contain a JSON object");
        return false;
    }

    impl_->config = doc.object();
    emit configLoaded(QString::fromStdString(path.string()));
    
    return true;
}

bool ConfigManager::saveConfig(const std::filesystem::path& path) {
    QJsonDocument doc(impl_->config);
    
    QFile file(QString::fromStdString(path.string()));
    if (!file.open(QIODevice::WriteOnly)) {
        emit configError(QString("Failed to open config file for writing: %1")
                        .arg(QString::fromStdString(path.string())));
        return false;
    }

    file.write(doc.toJson(QJsonDocument::Indented));
    file.close();

    emit configSaved(QString::fromStdString(path.string()));
    return true;
}

bool ConfigManager::loadDefaultConfig() {
    impl_->config = QJsonObject();

    // Application settings
    QJsonObject appSettings;
    appSettings["name"] = "DIREWOLF XAI";
    appSettings["version"] = "1.0.0";
    appSettings["log_level"] = "info";
    impl_->config["application"] = appSettings;

    // Voice settings
    QJsonObject voiceSettings;
    voiceSettings["recognition_engine"] = "whisper";
    voiceSettings["synthesis_engine"] = "azure";
    voiceSettings["language"] = "en-US";
    impl_->config["voice"] = voiceSettings;

    // NLP settings
    QJsonObject nlpSettings;
    nlpSettings["intent_threshold"] = 0.85;
    nlpSettings["entity_threshold"] = 0.80;
    impl_->config["nlp"] = nlpSettings;

    // Security settings
    QJsonObject securitySettings;
    securitySettings["drl_detection"] = true;
    securitySettings["malware_analysis"] = true;
    securitySettings["network_ids"] = true;
    securitySettings["auto_response"] = false;
    impl_->config["security"] = securitySettings;

    return true;
}

QVariant ConfigManager::getVariant(const std::string& key,
                                   const QVariant& defaultValue) const {
    QString qkey = QString::fromStdString(key);
    QStringList parts = qkey.split('.');

    QJsonValue value = impl_->config;
    for (const QString& part : parts) {
        if (!value.isObject()) {
            return defaultValue;
        }
        value = value.toObject()[part];
        if (value.isUndefined()) {
            return defaultValue;
        }
    }

    return value.toVariant();
}

void ConfigManager::setVariant(const std::string& key, const QVariant& value) {
    QString qkey = QString::fromStdString(key);
    QStringList parts = qkey.split('.');

    if (parts.isEmpty()) {
        return;
    }

    QJsonObject* current = &impl_->config;
    
    for (int i = 0; i < parts.size() - 1; ++i) {
        const QString& part = parts[i];
        
        if (!current->contains(part) || !(*current)[part].isObject()) {
            (*current)[part] = QJsonObject();
        }
        
        QJsonValue val = (*current)[part];
        QJsonObject obj = val.toObject();
        (*current)[part] = obj;
        current = &obj;
    }

    (*current)[parts.last()] = QJsonValue::fromVariant(value);
    
    emit valueChanged(qkey, value);
}

bool ConfigManager::hasKey(const std::string& key) const {
    QString qkey = QString::fromStdString(key);
    QStringList parts = qkey.split('.');

    QJsonValue value = impl_->config;
    for (const QString& part : parts) {
        if (!value.isObject()) {
            return false;
        }
        value = value.toObject()[part];
        if (value.isUndefined()) {
            return false;
        }
    }

    return true;
}

void ConfigManager::removeKey(const std::string& key) {
    QString qkey = QString::fromStdString(key);
    QStringList parts = qkey.split('.');

    if (parts.isEmpty()) {
        return;
    }

    QJsonObject* current = &impl_->config;
    
    for (int i = 0; i < parts.size() - 1; ++i) {
        const QString& part = parts[i];
        
        if (!current->contains(part) || !(*current)[part].isObject()) {
            return;
        }
        
        QJsonValue val = (*current)[part];
        QJsonObject obj = val.toObject();
        current = &obj;
    }

    current->remove(parts.last());
}

std::vector<std::string> ConfigManager::getAllKeys() const {
    std::vector<std::string> keys;
    
    std::function<void(const QJsonObject&, const QString&)> collectKeys;
    collectKeys = [&](const QJsonObject& obj, const QString& prefix) {
        for (auto it = obj.begin(); it != obj.end(); ++it) {
            QString key = prefix.isEmpty() ? it.key() : prefix + "." + it.key();
            
            if (it.value().isObject()) {
                collectKeys(it.value().toObject(), key);
            } else {
                keys.push_back(key.toStdString());
            }
        }
    };

    collectKeys(impl_->config, "");
    return keys;
}

std::vector<std::string> ConfigManager::getKeysInSection(const std::string& section) const {
    std::vector<std::string> keys;
    
    QString qsection = QString::fromStdString(section);
    QStringList parts = qsection.split('.');

    QJsonValue value = impl_->config;
    for (const QString& part : parts) {
        if (!value.isObject()) {
            return keys;
        }
        value = value.toObject()[part];
        if (value.isUndefined()) {
            return keys;
        }
    }

    if (value.isObject()) {
        QJsonObject obj = value.toObject();
        for (auto it = obj.begin(); it != obj.end(); ++it) {
            keys.push_back(it.key().toStdString());
        }
    }

    return keys;
}

void ConfigManager::beginSection(const std::string& section) {
    impl_->currentSection = section;
}

void ConfigManager::endSection() {
    impl_->currentSection.clear();
}

std::string ConfigManager::currentSection() const {
    return impl_->currentSection;
}

bool ConfigManager::validate() const {
    impl_->validationErrors.clear();
    
    // Add validation logic here
    // For now, just check if config is not empty
    if (impl_->config.isEmpty()) {
        impl_->validationErrors.push_back("Configuration is empty");
        return false;
    }

    return true;
}

std::vector<std::string> ConfigManager::getValidationErrors() const {
    return impl_->validationErrors;
}

} // namespace XAI
} // namespace DIREWOLF
