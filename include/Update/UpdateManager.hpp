/**
 * @file UpdateManager.hpp
 * @brief DIREWOLF Update Manager
 * 
 * Handles automatic updates with cryptographic verification and Alpha's permission.
 */

#pragma once

#include <QObject>
#include <QString>
#include <QDateTime>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <memory>
#include <vector>

namespace update {

/**
 * @brief Update channel types
 */
enum class UpdateChannel {
    STABLE,      // Production releases
    BETA,        // Pre-release testing
    DEVELOPMENT  // Latest features
};

/**
 * @brief Update status
 */
enum class UpdateStatus {
    IDLE,
    CHECKING,
    AVAILABLE,
    DOWNLOADING,
    VERIFYING,
    AWAITING_PERMISSION,
    INSTALLING,
    COMPLETE,
    FAILED,
    ROLLED_BACK
};

/**
 * @brief Update information
 */
struct UpdateInfo {
    QString version;
    QString channel;
    QString release_notes;
    QString download_url;
    QString signature_url;
    QString checksum;
    qint64 size_bytes;
    QDateTime release_date;
    bool requires_restart;
    bool is_critical;
    bool is_delta;
};

/**
 * @brief Update manifest
 */
struct UpdateManifest {
    QString version;
    QString channel;
    QDateTime timestamp;
    std::vector<UpdateInfo> updates;
    QString signature;
};

/**
 * @brief Update Manager
 * 
 * Manages automatic updates with:
 * - Background checking
 * - Cryptographic verification
 * - Alpha's permission requirement
 * - Automatic backup
 * - Rollback on failure
 * - Delta updates
 */
class UpdateManager : public QObject {
    Q_OBJECT
    
public:
    /**
     * @brief Constructor
     * @param parent Parent object
     */
    explicit UpdateManager(QObject* parent = nullptr);
    
    /**
     * @brief Destructor
     */
    ~UpdateManager();
    
    /**
     * @brief Initialize update manager
     * @param manifest_url URL to update manifest
     * @param public_key_path Path to public key for verification
     * @return True if successful
     */
    bool initialize(const QString& manifest_url, const QString& public_key_path);
    
    // ========== Update Checking ==========
    
    /**
     * @brief Check for updates
     * @param channel Update channel to check
     */
    void checkForUpdates(UpdateChannel channel = UpdateChannel::STABLE);
    
    /**
     * @brief Get current version
     * @return Current version string
     */
    QString getCurrentVersion() const;
    
    /**
     * @brief Get update channel
     * @return Current update channel
     */
    UpdateChannel getUpdateChannel() const;
    
    /**
     * @brief Set update channel
     * @param channel New update channel
     */
    void setUpdateChannel(UpdateChannel channel);
    
    /**
     * @brief Check if update is available
     * @return True if update available
     */
    bool isUpdateAvailable() const;
    
    /**
     * @brief Get available update info
     * @return Update information
     */
    UpdateInfo getAvailableUpdate() const;
    
    // ========== Update Download ==========
    
    /**
     * @brief Download update
     * @param update_info Update to download
     */
    void downloadUpdate(const UpdateInfo& update_info);
    
    /**
     * @brief Cancel download
     */
    void cancelDownload();
    
    /**
     * @brief Get download progress
     * @return Progress percentage (0-100)
     */
    int getDownloadProgress() const;
    
    // ========== Update Installation ==========
    
    /**
     * @brief Request permission to install update
     * @param update_info Update to install
     * @return Request ID for tracking
     */
    QString requestInstallPermission(const UpdateInfo& update_info);
    
    /**
     * @brief Install update (after permission granted)
     * @param backup_current Whether to backup current version
     * @return True if installation started
     */
    bool installUpdate(bool backup_current = true);
    
    /**
     * @brief Rollback to previous version
     * @return True if rollback successful
     */
    bool rollbackUpdate();
    
    // ========== Backup Management ==========
    
    /**
     * @brief Create backup of current installation
     * @return Backup path
     */
    QString createBackup();
    
    /**
     * @brief Restore from backup
     * @param backup_path Path to backup
     * @return True if successful
     */
    bool restoreBackup(const QString& backup_path);
    
    /**
     * @brief List available backups
     * @return List of backup paths
     */
    QStringList listBackups() const;
    
    /**
     * @brief Delete old backups
     * @param keep_count Number of backups to keep
     */
    void cleanupBackups(int keep_count = 3);
    
    // ========== Configuration ==========
    
    /**
     * @brief Enable/disable automatic updates
     * @param enabled Whether to enable
     */
    void setAutoUpdateEnabled(bool enabled);
    
    /**
     * @brief Check if auto-update is enabled
     * @return True if enabled
     */
    bool isAutoUpdateEnabled() const;
    
    /**
     * @brief Set update check frequency
     * @param hours Hours between checks
     */
    void setCheckFrequency(int hours);
    
    /**
     * @brief Get update status
     * @return Current status
     */
    UpdateStatus getStatus() const;
    
signals:
    /**
     * @brief Update available signal
     * @param update_info Available update
     */
    void updateAvailable(const UpdateInfo& update_info);
    
    /**
     * @brief Download progress signal
     * @param bytes_received Bytes downloaded
     * @param bytes_total Total bytes
     */
    void downloadProgress(qint64 bytes_received, qint64 bytes_total);
    
    /**
     * @brief Download complete signal
     */
    void downloadComplete();
    
    /**
     * @brief Verification complete signal
     * @param success Whether verification succeeded
     */
    void verificationComplete(bool success);
    
    /**
     * @brief Permission required signal
     * @param request_id Request identifier
     * @param update_info Update requiring permission
     */
    void permissionRequired(const QString& request_id, const UpdateInfo& update_info);
    
    /**
     * @brief Installation progress signal
     * @param progress Progress percentage
     */
    void installationProgress(int progress);
    
    /**
     * @brief Installation complete signal
     * @param success Whether installation succeeded
     */
    void installationComplete(bool success);
    
    /**
     * @brief Error signal
     * @param error_message Error description
     */
    void error(const QString& error_message);
    
    /**
     * @brief Status changed signal
     * @param status New status
     */
    void statusChanged(UpdateStatus status);
    
private slots:
    void onManifestDownloaded();
    void onUpdateDownloaded();
    void onDownloadProgress(qint64 received, qint64 total);
    void onNetworkError(QNetworkReply::NetworkError error);
    void onCheckTimerTimeout();
    
private:
    // Manifest operations
    bool downloadManifest();
    bool parseManifest(const QByteArray& data);
    bool verifyManifestSignature(const QByteArray& manifest, const QByteArray& signature);
    
    // Update operations
    bool verifyUpdateSignature(const QString& update_path, const QString& signature_path);
    bool verifyChecksum(const QString& file_path, const QString& expected_checksum);
    bool extractUpdate(const QString& update_path, const QString& extract_path);
    bool applyUpdate(const QString& update_path);
    
    // Delta updates
    bool isDeltaUpdate(const UpdateInfo& update_info);
    bool applyDeltaUpdate(const QString& delta_path);
    
    // Backup operations
    QString generateBackupPath();
    bool copyDirectory(const QString& source, const QString& destination);
    
    // Utility
    QString channelToString(UpdateChannel channel) const;
    UpdateChannel stringToChannel(const QString& channel) const;
    void setStatus(UpdateStatus status);
    QString getInstallationPath() const;
    
    // Data
    QString manifest_url_;
    QString public_key_path_;
    QString current_version_;
    UpdateChannel current_channel_;
    UpdateStatus current_status_;
    
    UpdateManifest current_manifest_;
    UpdateInfo available_update_;
    QString downloaded_update_path_;
    QString current_backup_path_;
    
    std::unique_ptr<QNetworkAccessManager> network_manager_;
    QNetworkReply* current_reply_;
    
    // Configuration
    bool auto_update_enabled_;
    int check_frequency_hours_;
    std::unique_ptr<QTimer> check_timer_;
    
    // Download tracking
    qint64 download_bytes_received_;
    qint64 download_bytes_total_;
    
    // Paths
    QString backup_directory_;
    QString download_directory_;
    QString temp_directory_;
};

} // namespace update
