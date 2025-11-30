#include "UI/GUIBackend.hpp"
#include <QTimer>
#include <QDebug>

namespace DIREWOLF {
namespace UI {

GUIBackend::GUIBackend(QObject* parent)
    : QObject(parent)
    , m_cpuUsage(0.0)
    , m_memoryUsage(0.0)
    , m_diskUsage(0.0)
    , m_threatsDetected(0)
    , m_systemStatus("Online")
    , m_isScanning(false)
{
    // Start update timer
    QTimer* updateTimer = new QTimer(this);
    connect(updateTimer, &QTimer::timeout, this, &GUIBackend::updateSystemMetrics);
    updateTimer->start(2000); // Update every 2 seconds
}

GUIBackend::~GUIBackend() = default;

void GUIBackend::startScan() {
    qDebug() << "Starting security scan...";
    m_isScanning = true;
    emit isScanningChanged();
    emit scanStarted();
    
    // Simulate scan completion after 5 seconds
    QTimer::singleShot(5000, this, [this]() {
        m_isScanning = false;
        emit isScanningChanged();
        emit scanCompleted(true, "No threats detected");
    });
}

void GUIBackend::stopScan() {
    qDebug() << "Stopping security scan...";
    m_isScanning = false;
    emit isScanningChanged();
    emit scanCompleted(false, "Scan cancelled by user");
}

void GUIBackend::executeVoiceCommand(const QString& command) {
    qDebug() << "Executing voice command:" << command;
    emit voiceCommandExecuted(command);
    
    // Process command
    QString response;
    if (command.contains("scan", Qt::CaseInsensitive)) {
        startScan();
        response = "Starting security scan now.";
    } else if (command.contains("status", Qt::CaseInsensitive)) {
        response = QString("System status: %1. CPU: %2%, Memory: %3%")
            .arg(m_systemStatus)
            .arg(m_cpuUsage, 0, 'f', 1)
            .arg(m_memoryUsage, 0, 'f', 1);
    } else {
        response = "Command received. Processing...";
    }
    
    emit voiceResponseReady(response);
}

void GUIBackend::updateSettings(const QString& key, const QVariant& value) {
    qDebug() << "Updating setting:" << key << "=" << value;
    m_settings[key] = value;
    emit settingsChanged();
}

QVariant GUIBackend::getSetting(const QString& key) const {
    return m_settings.value(key);
}

void GUIBackend::updateSystemMetrics() {
    // Simulate system metrics (in production, get real values)
    m_cpuUsage = 30.0 + (qrand() % 20);
    m_memoryUsage = 60.0 + (qrand() % 15);
    m_diskUsage = 40.0 + (qrand() % 10);
    
    emit cpuUsageChanged();
    emit memoryUsageChanged();
    emit diskUsageChanged();
    emit systemMetricsUpdated();
}

void GUIBackend::checkForThreats() {
    qDebug() << "Checking for threats...";
    // In production, integrate with actual threat detection
    emit threatCheckCompleted(0);
}

void GUIBackend::quarantineFile(const QString& filePath) {
    qDebug() << "Quarantining file:" << filePath;
    emit fileQuarantined(filePath, true);
}

void GUIBackend::restoreFile(const QString& filePath) {
    qDebug() << "Restoring file:" << filePath;
    emit fileRestored(filePath, true);
}

} // namespace UI
} // namespace DIREWOLF
