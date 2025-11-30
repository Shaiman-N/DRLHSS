#pragma once

#include <memory>
#include <string>
#include <functional>
#include <chrono>
#include <any>
#include <map>
#include <QObject>
#include <QVariant>

namespace DIREWOLF {
namespace XAI {

/**
 * @brief Event data structure
 */
struct Event {
    std::string type;
    std::chrono::system_clock::time_point timestamp;
    std::map<std::string, std::any> data;
    int priority = 0; // Higher priority events processed first

    Event() : timestamp(std::chrono::system_clock::now()) {}
    explicit Event(const std::string& eventType) 
        : type(eventType), timestamp(std::chrono::system_clock::now()) {}
};

/**
 * @brief Event handler callback type
 */
using EventHandler = std::function<void(const Event&)>;

/**
 * @brief Central event bus for application-wide event handling
 * 
 * Provides publish-subscribe pattern for loose coupling between components.
 * Thread-safe and supports priority-based event handling.
 */
class EventBus : public QObject {
    Q_OBJECT

public:
    explicit EventBus(QObject* parent = nullptr);
    ~EventBus() override;

    // Event publishing
    void publish(const Event& event);
    void publishAsync(const Event& event);
    
    // Event subscription
    int subscribe(const std::string& eventType, EventHandler handler);
    void unsubscribe(const std::string& eventType, int handlerId);
    void unsubscribeAll(const std::string& eventType);

    // Event filtering
    void setEventFilter(const std::string& eventType, 
                       std::function<bool(const Event&)> filter);
    void removeEventFilter(const std::string& eventType);

    // Statistics
    size_t getSubscriberCount(const std::string& eventType) const;
    size_t getTotalEventCount() const;
    void clearStatistics();

signals:
    void eventPublished(const QString& eventType);
    void eventProcessed(const QString& eventType, int handlerCount);
    void eventError(const QString& eventType, const QString& error);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace XAI
} // namespace DIREWOLF
