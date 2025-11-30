#pragma once

#include <memory>
#include <string>
#include <filesystem>
#include <QObject>

namespace DIREWOLF {
namespace XAI {

/**
 * @brief Log levels
 */
enum class LogLevel {
    Trace,
    Debug,
    Info,
    Warning,
    Error,
    Critical
};

/**
 * @brief Logging system
 * 
 * Provides structured logging with multiple outputs (console, file, Qt).
 * Thread-safe and supports log rotation.
 */
class Logger : public QObject {
    Q_OBJECT

public:
    explicit Logger(QObject* parent = nullptr);
    ~Logger() override;

    // Initialization
    bool initialize(const std::filesystem::path& logFile, LogLevel level = LogLevel::Info);
    void shutdown();

    // Logging methods
    void trace(const std::string& message, const std::string& category = "");
    void debug(const std::string& message, const std::string& category = "");
    void info(const std::string& message, const std::string& category = "");
    void warning(const std::string& message, const std::string& category = "");
    void error(const std::string& message, const std::string& category = "");
    void critical(const std::string& message, const std::string& category = "");

    // Configuration
    void setLogLevel(LogLevel level);
    LogLevel getLogLevel() const;
    void setLogFile(const std::filesystem::path& path);
    void enableConsoleOutput(bool enable);
    void enableFileOutput(bool enable);

    // Log rotation
    void setMaxFileSize(size_t bytes);
    void setMaxFiles(size_t count);
    void rotateLog();

signals:
    void logMessage(int level, const QString& message, const QString& category);
    void logError(const QString& error);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    void log(LogLevel level, const std::string& message, const std::string& category);
};

// Convenience macros
#define LOG_TRACE(logger, msg) if(logger) logger->trace(msg, __FUNCTION__)
#define LOG_DEBUG(logger, msg) if(logger) logger->debug(msg, __FUNCTION__)
#define LOG_INFO(logger, msg) if(logger) logger->info(msg, __FUNCTION__)
#define LOG_WARNING(logger, msg) if(logger) logger->warning(msg, __FUNCTION__)
#define LOG_ERROR(logger, msg) if(logger) logger->error(msg, __FUNCTION__)
#define LOG_CRITICAL(logger, msg) if(logger) logger->critical(msg, __FUNCTION__)

} // namespace XAI
} // namespace DIREWOLF
