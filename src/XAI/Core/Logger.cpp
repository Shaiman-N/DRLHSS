#include "XAI/Core/Logger.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <mutex>
#include <chrono>
#include <QDateTime>
#include <QDir>

namespace DIREWOLF {
namespace XAI {

struct Logger::Impl {
    std::ofstream logFile;
    LogLevel currentLevel = LogLevel::Info;
    bool consoleOutput = true;
    bool fileOutput = true;
    size_t maxFileSize = 10 * 1024 * 1024; // 10MB
    size_t maxFiles = 5;
    std::filesystem::path logFilePath;
    mutable std::mutex mutex;
};

Logger::Logger(QObject* parent)
    : QObject(parent)
    , impl_(std::make_unique<Impl>())
{
}

Logger::~Logger() {
    shutdown();
}

bool Logger::initialize(const std::filesystem::path& logFile, LogLevel level) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    impl_->logFilePath = logFile;
    impl_->currentLevel = level;

    // Create log directory if it doesn't exist
    auto logDir = logFile.parent_path();
    if (!std::filesystem::exists(logDir)) {
        std::filesystem::create_directories(logDir);
    }

    // Open log file
    impl_->logFile.open(logFile, std::ios::app);
    if (!impl_->logFile.is_open()) {
        std::cerr << "Failed to open log file: " << logFile << std::endl;
        return false;
    }

    // Write startup message
    log(LogLevel::Info, "Logger initialized", "Logger");
    
    return true;
}

void Logger::shutdown() {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    if (impl_->logFile.is_open()) {
        log(LogLevel::Info, "Logger shutting down", "Logger");
        impl_->logFile.close();
    }
}

void Logger::trace(const std::string& message, const std::string& category) {
    log(LogLevel::Trace, message, category);
}

void Logger::debug(const std::string& message, const std::string& category) {
    log(LogLevel::Debug, message, category);
}

void Logger::info(const std::string& message, const std::string& category) {
    log(LogLevel::Info, message, category);
}

void Logger::warning(const std::string& message, const std::string& category) {
    log(LogLevel::Warning, message, category);
}

void Logger::error(const std::string& message, const std::string& category) {
    log(LogLevel::Error, message, category);
}

void Logger::critical(const std::string& message, const std::string& category) {
    log(LogLevel::Critical, message, category);
}

void Logger::log(LogLevel level, const std::string& message, const std::string& category) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    // Check log level
    if (level < impl_->currentLevel) {
        return;
    }

    // Format timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();

    // Format level
    std::string levelStr;
    switch (level) {
        case LogLevel::Trace:    levelStr = "TRACE"; break;
        case LogLevel::Debug:    levelStr = "DEBUG"; break;
        case LogLevel::Info:     levelStr = "INFO "; break;
        case LogLevel::Warning:  levelStr = "WARN "; break;
        case LogLevel::Error:    levelStr = "ERROR"; break;
        case LogLevel::Critical: levelStr = "CRIT "; break;
    }

    // Format log message
    std::string logMessage = ss.str() + " [" + levelStr + "]";
    if (!category.empty()) {
        logMessage += " [" + category + "]";
    }
    logMessage += " " + message;

    // Output to console
    if (impl_->consoleOutput) {
        if (level >= LogLevel::Error) {
            std::cerr << logMessage << std::endl;
        } else {
            std::cout << logMessage << std::endl;
        }
    }

    // Output to file
    if (impl_->fileOutput && impl_->logFile.is_open()) {
        impl_->logFile << logMessage << std::endl;
        impl_->logFile.flush();

        // Check file size for rotation
        auto currentSize = impl_->logFile.tellp();
        if (static_cast<size_t>(currentSize) >= impl_->maxFileSize) {
            rotateLog();
        }
    }

    // Emit Qt signal
    emit logMessage(static_cast<int>(level), 
                   QString::fromStdString(message),
                   QString::fromStdString(category));
}

void Logger::setLogLevel(LogLevel level) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->currentLevel = level;
}

LogLevel Logger::getLogLevel() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->currentLevel;
}

void Logger::setLogFile(const std::filesystem::path& path) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    if (impl_->logFile.is_open()) {
        impl_->logFile.close();
    }

    impl_->logFilePath = path;
    impl_->logFile.open(path, std::ios::app);
}

void Logger::enableConsoleOutput(bool enable) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->consoleOutput = enable;
}

void Logger::enableFileOutput(bool enable) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->fileOutput = enable;
}

void Logger::setMaxFileSize(size_t bytes) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->maxFileSize = bytes;
}

void Logger::setMaxFiles(size_t count) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->maxFiles = count;
}

void Logger::rotateLog() {
    if (!impl_->logFile.is_open()) {
        return;
    }

    impl_->logFile.close();

    // Rotate existing log files
    for (size_t i = impl_->maxFiles - 1; i > 0; --i) {
        auto oldPath = impl_->logFilePath.string() + "." + std::to_string(i);
        auto newPath = impl_->logFilePath.string() + "." + std::to_string(i + 1);
        
        if (std::filesystem::exists(oldPath)) {
            if (i == impl_->maxFiles - 1) {
                std::filesystem::remove(oldPath);
            } else {
                std::filesystem::rename(oldPath, newPath);
            }
        }
    }

    // Rename current log file
    if (std::filesystem::exists(impl_->logFilePath)) {
        auto backupPath = impl_->logFilePath.string() + ".1";
        std::filesystem::rename(impl_->logFilePath, backupPath);
    }

    // Open new log file
    impl_->logFile.open(impl_->logFilePath, std::ios::app);
}

} // namespace XAI
} // namespace DIREWOLF
