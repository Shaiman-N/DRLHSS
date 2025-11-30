/**
 * @file UpdateManager.cpp
 * @brief DIREWOLF Update Manager Implementation
 */

#include "Update/UpdateManager.hpp"
#include <QFile>
#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QCryptographicHash>
#include <QProcess>
#include <QTimer>
#include <QDebug>

namespace update {

UpdateManager::UpdateManager(QObject* parent)
    : QObject(parent)
    , current_version_("1.0.0")
    , current_channel_(UpdateChannel::STABLE)
    , current_status_(UpdateStatus::IDLE)
    , current_reply_(nullptr)
    , auto_update_enabled_(true)
    , check_frequency_hours_(24)
    , download_bytes_received_(0)
    , download_bytes_total_(0)
{
    network_manager_ = std::make_unique<QNetworkAccessManager>(this);
    check_timer_ = std::make_unique<QTimer>(this);
    
    connect(check_timer_.get(), &QTimer::timeout, this, &UpdateManager::onCheckTimerTimeout);
}

UpdateManager::~UpdateManager() {
    if (current_reply_) {
        current_reply_->abort();
        current_reply_->deleteLater();
    }
}

bool UpdateManager::initialize(const QString& manifest_url, const QString& public_key_path) {
    manifest_url_ = manifest_url;
    public_key_path_ = public_key_path;
    
    // Setup directories
    QString app_data = QDir::homePath() + "/.direwolf";
    backup_directory_ = app_data + "/backups";
    download_directory_ = app_data + "/downloads";
    temp_directory_ = app_data + "/temp";
    
    QDir().mkpath(backup_directory_);
    QDir().mkpath(download_directory_);
    QDir().mkpath(temp_directory_);
    
    // Verify public key exists
    if (!QFile::exists(public_key_path_)) {
        qWarning() << "Public key not found:" << public_key_path_;
        return false;
    }
    
    // Start automatic check timer
    if (auto_update_enabled_) {
        check_timer_->start(check_frequency_hours_ * 3600 * 1000);
    }
    
    qInfo() << "Update Manager initialized";
    qInfo() << "Current version:" << current_version_;
    qInfo() << "Update channel:" << channelToString(current_channel_);
    
    return true;
}

void UpdateManager::checkForUpdates(UpdateChannel channel) {
    if (current_status_ == UpdateStatus::CHECKING) {
        qWarning() << "Already checking for updates";
        return;
    }
    
    setStatus(UpdateStatus::CHECKING);
    current_channel_ = channel;
    
    qInfo() << "Checking for updates on channel:" << channelToString(channel);
    
    downloadManifest();
}

QString UpdateManager::getCurrentVersion() const {
    return current_version_;
}

UpdateChannel UpdateManager::getUpdateChannel() const {
    return current_channel_;
}

void UpdateManager::setUpdateChannel(UpdateChannel channel) {
    current_channel_ = channel;
    qInfo() << "Update channel changed to:" << channelToString(channel);
}

bool UpdateManager::isUpdateAvailable() const {
    return !available_update_.version.isEmpty() &&
           available_update_.version != current_version_;
}

UpdateInfo UpdateManager::getAvailableUpdate() const {
    return available_update_;
}

void UpdateManager::downloadUpdate(const UpdateInfo& update_info) {
    if (current_status_ == UpdateStatus::DOWNLOADING) {
        qWarning() << "Already downloading update";
        return;
    }
    
    setStatus(UpdateStatus::DOWNLOADING);
    available_update_ = update_info;
    
    qInfo() << "Downloading update:" << update_info.version;
    qInfo() << "Size:" << update_info.size_bytes << "bytes";
    
    // Download update file
    QNetworkRequest request(update_info.download_url);
    current_reply_ = network_manager_->get(request);
    
    connect(current_reply_, &QNetworkReply::downloadProgress,
            this, &UpdateManager::onDownloadProgress);
    connect(current_reply_, &QNetworkReply::finished,
            this, &UpdateManager::onUpdateDownloaded);
    connect(current_reply_, QOverload<QNetworkReply::NetworkError>::of(&QNetworkReply::error),
            this, &UpdateManager::onNetworkError);
}

void UpdateManager::cancelDownload() {
    if (current_reply_) {
        current_reply_->abort();
        current_reply_->deleteLater();
        current_reply_ = nullptr;
    }
    
    setStatus(UpdateStatus::IDLE);
    qInfo() << "Download cancelled";
}

int UpdateManager::getDownloadProgress() const {
    if (download_bytes_total_ == 0) return 0;
    return (download_bytes_received_ * 100) / download_bytes_total_;
}

QString UpdateManager::requestInstallPermission(const UpdateInfo& update_info) {
    QString request_id = QDateTime::currentDateTime().toString("yyyyMMddHHmmss");
    
    setStatus(UpdateStatus::AWAITING_PERMISSION);
    
    qInfo() << "Requesting permission to install update:" << update_info.version;
    emit permissionRequired(request_id, update_info);
    
    return request_id;
}

bool UpdateManager::installUpdate(bool backup_current) {
    if (downloaded_update_path_.isEmpty()) {
        qWarning() << "No update downloaded";
        return false;
    }
    
    setStatus(UpdateStatus::INSTALLING);
    
    qInfo() << "Installing update...";
    
    // Create backup if requested
    if (backup_current) {
        qInfo() << "Creating backup...";
        current_backup_path_ = createBackup();
        
        if (current_backup_path_.isEmpty()) {
            qWarning() << "Backup failed";
            emit error("Failed to create backup");
            setStatus(UpdateStatus::FAILED);
            return false;
        }
        
        qInfo() << "Backup created:" << current_backup_path_;
    }
    
    // Apply update
    bool success = applyUpdate(downloaded_update_path_);
    
    if (success) {
        qInfo() << "Update installed successfully";
        setStatus(UpdateStatus::COMPLETE);
        emit installationComplete(true);
    } else {
        qWarning() << "Update installation failed";
        
        // Attempt rollback
        if (!current_backup_path_.isEmpty()) {
            qInfo() << "Attempting rollback...";
            if (restoreBackup(current_backup_path_)) {
                qInfo() << "Rollback successful";
                setStatus(UpdateStatus::ROLLED_BACK);
            } else {
                qWarning() << "Rollback failed";
                setStatus(UpdateStatus::FAILED);
            }
        } else {
            setStatus(UpdateStatus::FAILED);
        }
        
        emit installationComplete(false);
    }
    
    return success;
}

bool UpdateManager::rollbackUpdate() {
    if (current_backup_path_.isEmpty()) {
        qWarning() << "No backup available for rollback";
        return false;
    }
    
    qInfo() << "Rolling back to backup:" << current_backup_path_;
    
    bool success = restoreBackup(current_backup_path_);
    
    if (success) {
        qInfo() << "Rollback successful";
        setStatus(UpdateStatus::ROLLED_BACK);
    } else {
        qWarning() << "Rollback failed";
        setStatus(UpdateStatus::FAILED);
    }
    
    return success;
}

QString UpdateManager::createBackup() {
    QString backup_path = generateBackupPath();
    QString install_path = getInstallationPath();
    
    qInfo() << "Creating backup from:" << install_path;
    qInfo() << "Backup destination:" << backup_path;
    
    if (copyDirectory(install_path, backup_path)) {
        return backup_path;
    }
    
    return QString();
}

bool UpdateManager::restoreBackup(const QString& backup_path) {
    if (!QDir(backup_path).exists()) {
        qWarning() << "Backup not found:" << backup_path;
        return false;
    }
    
    QString install_path = getInstallationPath();
    
    qInfo() << "Restoring from backup:" << backup_path;
    qInfo() << "Restore destination:" << install_path;
    
    // Remove current installation
    QDir install_dir(install_path);
    if (!install_dir.removeRecursively()) {
        qWarning() << "Failed to remove current installation";
        return false;
    }
    
    // Restore from backup
    return copyDirectory(backup_path, install_path);
}

QStringList UpdateManager::listBackups() const {
    QDir backup_dir(backup_directory_);
    QStringList backups = backup_dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Time);
    
    QStringList full_paths;
    for (const QString& backup : backups) {
        full_paths.append(backup_dir.absoluteFilePath(backup));
    }
    
    return full_paths;
}

void UpdateManager::cleanupBackups(int keep_count) {
    QStringList backups = listBackups();
    
    // Remove old backups
    for (int i = keep_count; i < backups.size(); ++i) {
        QDir backup_dir(backups[i]);
        if (backup_dir.removeRecursively()) {
            qInfo() << "Removed old backup:" << backups[i];
        }
    }
}

void UpdateManager::setAutoUpdateEnabled(bool enabled) {
    auto_update_enabled_ = enabled;
    
    if (enabled) {
        check_timer_->start(check_frequency_hours_ * 3600 * 1000);
        qInfo() << "Automatic updates enabled";
    } else {
        check_timer_->stop();
        qInfo() << "Automatic updates disabled";
    }
}

bool UpdateManager::isAutoUpdateEnabled() const {
    return auto_update_enabled_;
}

void UpdateManager::setCheckFrequency(int hours) {
    check_frequency_hours_ = hours;
    
    if (auto_update_enabled_) {
        check_timer_->setInterval(hours * 3600 * 1000);
    }
    
    qInfo() << "Update check frequency set to:" << hours << "hours";
}

UpdateStatus UpdateManager::getStatus() const {
    return current_status_;
}

void UpdateManager::onManifestDownloaded() {
    if (!current_reply_) return;
    
    QByteArray data = current_reply_->readAll();
    current_reply_->deleteLater();
    current_reply_ = nullptr;
    
    if (parseManifest(data)) {
        qInfo() << "Manifest parsed successfully";
        
        // Check if update available
        for (const auto& update : current_manifest_.updates) {
            if (update.version != current_version_ &&
                update.channel == channelToString(current_channel_)) {
                
                available_update_ = update;
                setStatus(UpdateStatus::AVAILABLE);
                emit updateAvailable(update);
                
                qInfo() << "Update available:" << update.version;
                return;
            }
        }
        
        qInfo() << "No updates available";
        setStatus(UpdateStatus::IDLE);
    } else {
        qWarning() << "Failed to parse manifest";
        emit error("Failed to parse update manifest");
        setStatus(UpdateStatus::FAILED);
    }
}

void UpdateManager::onUpdateDownloaded() {
    if (!current_reply_) return;
    
    QByteArray data = current_reply_->readAll();
    current_reply_->deleteLater();
    current_reply_ = nullptr;
    
    // Save to file
    downloaded_update_path_ = download_directory_ + "/update_" + 
                              available_update_.version + ".zip";
    
    QFile file(downloaded_update_path_);
    if (!file.open(QIODevice::WriteOnly)) {
        qWarning() << "Failed to save update file";
        emit error("Failed to save update file");
        setStatus(UpdateStatus::FAILED);
        return;
    }
    
    file.write(data);
    file.close();
    
    qInfo() << "Update downloaded:" << downloaded_update_path_;
    emit downloadComplete();
    
    // Verify checksum
    setStatus(UpdateStatus::VERIFYING);
    
    if (verifyChecksum(downloaded_update_path_, available_update_.checksum)) {
        qInfo() << "Checksum verified";
        emit verificationComplete(true);
        
        // Request permission to install
        requestInstallPermission(available_update_);
    } else {
        qWarning() << "Checksum verification failed";
        emit verificationComplete(false);
        emit error("Update verification failed");
        setStatus(UpdateStatus::FAILED);
    }
}

void UpdateManager::onDownloadProgress(qint64 received, qint64 total) {
    download_bytes_received_ = received;
    download_bytes_total_ = total;
    
    emit downloadProgress(received, total);
}

void UpdateManager::onNetworkError(QNetworkReply::NetworkError error) {
    qWarning() << "Network error:" << error;
    emit this->error("Network error during update");
    setStatus(UpdateStatus::FAILED);
}

void UpdateManager::onCheckTimerTimeout() {
    if (auto_update_enabled_) {
        qInfo() << "Automatic update check triggered";
        checkForUpdates(current_channel_);
    }
}

bool UpdateManager::downloadManifest() {
    QNetworkRequest request(manifest_url_);
    current_reply_ = network_manager_->get(request);
    
    connect(current_reply_, &QNetworkReply::finished,
            this, &UpdateManager::onManifestDownloaded);
    connect(current_reply_, QOverload<QNetworkReply::NetworkError>::of(&QNetworkReply::error),
            this, &UpdateManager::onNetworkError);
    
    return true;
}

bool UpdateManager::parseManifest(const QByteArray& data) {
    QJsonDocument doc = QJsonDocument::fromJson(data);
    
    if (!doc.isObject()) {
        return false;
    }
    
    QJsonObject root = doc.object();
    
    current_manifest_.version = root["version"].toString();
    current_manifest_.channel = root["channel"].toString();
    current_manifest_.timestamp = QDateTime::fromString(
        root["timestamp"].toString(), Qt::ISODate);
    
    QJsonArray updates = root["updates"].toArray();
    current_manifest_.updates.clear();
    
    for (const QJsonValue& value : updates) {
        QJsonObject obj = value.toObject();
        
        UpdateInfo info;
        info.version = obj["version"].toString();
        info.channel = obj["channel"].toString();
        info.release_notes = obj["release_notes"].toString();
        info.download_url = obj["download_url"].toString();
        info.signature_url = obj["signature_url"].toString();
        info.checksum = obj["checksum"].toString();
        info.size_bytes = obj["size_bytes"].toVariant().toLongLong();
        info.release_date = QDateTime::fromString(
            obj["release_date"].toString(), Qt::ISODate);
        info.requires_restart = obj["requires_restart"].toBool();
        info.is_critical = obj["is_critical"].toBool();
        info.is_delta = obj["is_delta"].toBool();
        
        current_manifest_.updates.push_back(info);
    }
    
    return true;
}

bool UpdateManager::verifyChecksum(const QString& file_path, const QString& expected_checksum) {
    QFile file(file_path);
    if (!file.open(QIODevice::ReadOnly)) {
        return false;
    }
    
    QCryptographicHash hash(QCryptographicHash::Sha256);
    hash.addData(&file);
    
    QString actual_checksum = hash.result().toHex();
    
    return actual_checksum == expected_checksum;
}

bool UpdateManager::applyUpdate(const QString& update_path) {
    // Extract update
    QString extract_path = temp_directory_ + "/update_extract";
    
    if (!extractUpdate(update_path, extract_path)) {
        return false;
    }
    
    // Copy files to installation directory
    QString install_path = getInstallationPath();
    
    return copyDirectory(extract_path, install_path);
}

bool UpdateManager::extractUpdate(const QString& update_path, const QString& extract_path) {
    // Use system unzip command
    QProcess process;
    process.start("unzip", QStringList() << "-o" << update_path << "-d" << extract_path);
    process.waitForFinished();
    
    return process.exitCode() == 0;
}

bool UpdateManager::copyDirectory(const QString& source, const QString& destination) {
    QDir source_dir(source);
    QDir dest_dir(destination);
    
    if (!dest_dir.exists()) {
        dest_dir.mkpath(".");
    }
    
    QFileInfoList entries = source_dir.entryInfoList(
        QDir::Files | QDir::Dirs | QDir::NoDotAndDotDot);
    
    for (const QFileInfo& entry : entries) {
        QString dest_path = destination + "/" + entry.fileName();
        
        if (entry.isDir()) {
            if (!copyDirectory(entry.absoluteFilePath(), dest_path)) {
                return false;
            }
        } else {
            if (!QFile::copy(entry.absoluteFilePath(), dest_path)) {
                return false;
            }
        }
    }
    
    return true;
}

QString UpdateManager::generateBackupPath() {
    QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
    return backup_directory_ + "/backup_" + current_version_ + "_" + timestamp;
}

QString UpdateManager::channelToString(UpdateChannel channel) const {
    switch (channel) {
        case UpdateChannel::STABLE: return "stable";
        case UpdateChannel::BETA: return "beta";
        case UpdateChannel::DEVELOPMENT: return "development";
        default: return "stable";
    }
}

UpdateChannel UpdateManager::stringToChannel(const QString& channel) const {
    if (channel == "beta") return UpdateChannel::BETA;
    if (channel == "development") return UpdateChannel::DEVELOPMENT;
    return UpdateChannel::STABLE;
}

void UpdateManager::setStatus(UpdateStatus status) {
    if (current_status_ != status) {
        current_status_ = status;
        emit statusChanged(status);
    }
}

QString UpdateManager::getInstallationPath() const {
    // Return application installation directory
    return QCoreApplication::applicationDirPath();
}

} // namespace update
