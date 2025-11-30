#pragma once

#include <QObject>
#include <QString>
#include <QVariant>
#include <QMap>

namespace DIREWOLF {
namespace UI {

class GUIBackend : public QObject {
    Q_OBJECT
    Q_PROPERTY(double cpuUsage READ cpuUsage NOTIFY cpuUsageChanged)
    Q_PROPERTY(double memoryUsage READ memoryUsage NOTIFY memoryUsageChanged)
    Q_PROPERTY(double diskUsage READ diskUsage NOTIFY diskUsageChanged)
    Q_PROPERTY(int threatsDetected READ threatsDetected NOTIFY threatsDetectedChanged)
    Q_PROPERTY(QString systemStatus READ systemStatus NOTIFY systemStatusChanged)
    Q_PROPERTY(bool isScanning READ isScanning NOTIFY isScanningChanged)

public:
    explicit GUIBackend(QObject* parent = nullptr);
    ~GUIBackend();

    // Property getters
    double cpuUsage() const { return m_cpuUsage; }
    double memoryUsage() const { return m_memoryUsage; }
    double diskUsage() const { return m_diskUsage; }
    int threatsDetected() const { return m_threatsDetected; }
    QString systemStatus() const { return m_systemStatus; }
    bool isScanning() const { return m_isScanning; }

public slots:
    // Security operations
    void startScan();
    void stopScan();
    void checkForThreats();
    void quarantineFile(const QString& filePath);
    void restoreFile(const QString& filePath);

    // Voice commands
    void executeVoiceCommand(const QString& command);

    // Settings
    void updateSettings(const QString& key, const QVariant& value);
    QVariant getSetting(const QString& key) const;

signals:
    // Property change signals
    void cpuUsageChanged();
    void memoryUsageChanged();
    void diskUsageChanged();
    void threatsDetectedChanged();
    void systemStatusChanged();
    void isScanningChanged();

    // Operation signals
    void scanStarted();
    void scanCompleted(bool success, const QString& message);
    void threatCheckCompleted(int threatsFound);
    void fileQuarantined(const QString& filePath, bool success);
    void fileRestored(const QString& filePath, bool success);

    // Voice signals
    void voiceCommandExecuted(const QString& command);
    void voiceResponseReady(const QString& response);

    // System signals
    void systemMetricsUpdated();
    void settingsChanged();

private:
    void updateSystemMetrics();

    double m_cpuUsage;
    double m_memoryUsage;
    double m_diskUsage;
    int m_threatsDetected;
    QString m_systemStatus;
    bool m_isScanning;
    QMap<QString, QVariant> m_settings;
};

} // namespace UI
} // namespace DIREWOLF
