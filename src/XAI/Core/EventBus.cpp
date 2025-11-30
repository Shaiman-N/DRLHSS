#include "XAI/Core/EventBus.hpp"

#include <map>
#include <vector>
#include <mutex>
#include <queue>
#include <atomic>
#include <QThreadPool>
#include <QRunnable>

namespace DIREWOLF {
namespace XAI {

struct EventBus::Impl {
    struct HandlerInfo {
        int id;
        EventHandler handler;
    };

    std::map<std::string, std::vector<HandlerInfo>> handlers;
    std::map<std::string, std::function<bool(const Event&)>> filters;
    std::atomic<int> nextHandlerId{0};
    std::atomic<size_t> totalEventCount{0};
    mutable std::mutex mutex;
};

class EventTask : public QRunnable {
public:
    EventTask(Event event, std::vector<EventBus::Impl::HandlerInfo> handlers)
        : event_(std::move(event)), handlers_(std::move(handlers)) {}

    void run() override {
        for (const auto& handlerInfo : handlers_) {
            try {
                handlerInfo.handler(event_);
            } catch (const std::exception& e) {
                // Log error but continue processing
            }
        }
    }

private:
    Event event_;
    std::vector<EventBus::Impl::HandlerInfo> handlers_;
};

EventBus::EventBus(QObject* parent)
    : QObject(parent)
    , impl_(std::make_unique<Impl>())
{
}

EventBus::~EventBus() = default;

void EventBus::publish(const Event& event) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    impl_->totalEventCount++;

    // Check if there are subscribers
    auto it = impl_->handlers.find(event.type);
    if (it == impl_->handlers.end() || it->second.empty()) {
        return;
    }

    // Apply filter if exists
    auto filterIt = impl_->filters.find(event.type);
    if (filterIt != impl_->filters.end()) {
        try {
            if (!filterIt->second(event)) {
                return; // Event filtered out
            }
        } catch (const std::exception& e) {
            emit eventError(QString::fromStdString(event.type),
                          QString("Filter error: %1").arg(e.what()));
            return;
        }
    }

    // Call handlers synchronously
    int handlerCount = 0;
    for (const auto& handlerInfo : it->second) {
        try {
            handlerInfo.handler(event);
            handlerCount++;
        } catch (const std::exception& e) {
            emit eventError(QString::fromStdString(event.type),
                          QString("Handler error: %1").arg(e.what()));
        }
    }

    emit eventPublished(QString::fromStdString(event.type));
    emit eventProcessed(QString::fromStdString(event.type), handlerCount);
}

void EventBus::publishAsync(const Event& event) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    impl_->totalEventCount++;

    // Check if there are subscribers
    auto it = impl_->handlers.find(event.type);
    if (it == impl_->handlers.end() || it->second.empty()) {
        return;
    }

    // Apply filter if exists
    auto filterIt = impl_->filters.find(event.type);
    if (filterIt != impl_->filters.end()) {
        try {
            if (!filterIt->second(event)) {
                return; // Event filtered out
            }
        } catch (const std::exception& e) {
            emit eventError(QString::fromStdString(event.type),
                          QString("Filter error: %1").arg(e.what()));
            return;
        }
    }

    // Create copy of handlers for async execution
    std::vector<Impl::HandlerInfo> handlersCopy = it->second;
    
    // Schedule async execution
    auto* task = new EventTask(event, std::move(handlersCopy));
    task->setAutoDelete(true);
    QThreadPool::globalInstance()->start(task);

    emit eventPublished(QString::fromStdString(event.type));
}

int EventBus::subscribe(const std::string& eventType, EventHandler handler) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    int handlerId = impl_->nextHandlerId++;
    
    Impl::HandlerInfo info;
    info.id = handlerId;
    info.handler = std::move(handler);

    impl_->handlers[eventType].push_back(std::move(info));

    return handlerId;
}

void EventBus::unsubscribe(const std::string& eventType, int handlerId) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    auto it = impl_->handlers.find(eventType);
    if (it == impl_->handlers.end()) {
        return;
    }

    auto& handlers = it->second;
    handlers.erase(
        std::remove_if(handlers.begin(), handlers.end(),
                      [handlerId](const Impl::HandlerInfo& info) {
                          return info.id == handlerId;
                      }),
        handlers.end());

    // Remove event type if no more handlers
    if (handlers.empty()) {
        impl_->handlers.erase(it);
    }
}

void EventBus::unsubscribeAll(const std::string& eventType) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->handlers.erase(eventType);
}

void EventBus::setEventFilter(const std::string& eventType,
                              std::function<bool(const Event&)> filter) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->filters[eventType] = std::move(filter);
}

void EventBus::removeEventFilter(const std::string& eventType) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->filters.erase(eventType);
}

size_t EventBus::getSubscriberCount(const std::string& eventType) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    auto it = impl_->handlers.find(eventType);
    if (it != impl_->handlers.end()) {
        return it->second.size();
    }
    
    return 0;
}

size_t EventBus::getTotalEventCount() const {
    return impl_->totalEventCount.load();
}

void EventBus::clearStatistics() {
    impl_->totalEventCount = 0;
}

} // namespace XAI
} // namespace DIREWOLF
