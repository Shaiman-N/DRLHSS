#include "XAI/Voice/VoiceInterface.hpp"
#include "XAI/Core/EventBus.hpp"
#include "XAI/Core/Logger.hpp"

#include <QTextToSpeech>
#include <QTimer>
#include <thread>

namespace DIREWOLF {
namespace XAI {

struct VoiceInterface::Impl {
    EventBus* eventBus = nullptr;
    Logger* logger = nullptr;
    std::unique_ptr<QTextToSpeech> tts;
    bool listening = false;
    bool speaking = false;
    std::string language = "en-US";
    std::string recognitionEngine = "system";
    std::string synthesisEngine = "qt";
};

VoiceInterface::VoiceInterface(EventBus* eventBus, Logger* logger, QObject* parent)
    : QObject(parent)
    , impl_(std::make_unique<Impl>())
{
    impl_->eventBus = eventBus;
    impl_->logger = logger;
}

VoiceInterface::~VoiceInterface() {
    shutdown();
}

bool VoiceInterface::initialize() {
    LOG_INFO(impl_->logger, "Initializing voice interface");

    // Initialize TTS
    impl_->tts = std::make_unique<QTextToSpeech>();
    
    connect(impl_->tts.get(), &QTextToSpeech::stateChanged,
            [this](QTextToSpeech::State state) {
        if (state == QTextToSpeech::Speaking) {
            impl_->speaking = true;
        } else if (state == QTextToSpeech::Ready) {
            if (impl_->speaking) {
                impl_->speaking = false;
                emit speakingFinished();
                
                Event event("voice.speaking.finished");
                impl_->eventBus->publish(event);
            }
        }
    });

    LOG_INFO(impl_->logger, "Voice interface initialized");
    return true;
}

void VoiceInterface::shutdown() {
    stopListening();
    stopSpeaking();
    impl_->tts.reset();
}

void VoiceInterface::startListening() {
    if (impl_->listening) {
        return;
    }

    LOG_INFO(impl_->logger, "Starting voice recognition");
    impl_->listening = true;
    emit listeningStarted();

    Event event("voice.listening.started");
    impl_->eventBus->publish(event);

    // Simulate recognition for demo (replace with actual Whisper integration)
    QTimer::singleShot(2000, this, [this]() {
        if (impl_->listening) {
            std::string recognizedText = "scan for threats";
            float confidence = 0.95f;
            
            LOG_INFO(impl_->logger, "Speech recognized: " + recognizedText);
            emit speechRecognized(QString::fromStdString(recognizedText), confidence);
            
            Event event("voice.speech.recognized");
            event.data["text"] = recognizedText;
            event.data["confidence"] = confidence;
            impl_->eventBus->publish(event);
        }
    });
}

void VoiceInterface::stopListening() {
    if (!impl_->listening) {
        return;
    }

    LOG_INFO(impl_->logger, "Stopping voice recognition");
    impl_->listening = false;
    emit listeningStopped();

    Event event("voice.listening.stopped");
    impl_->eventBus->publish(event);
}

bool VoiceInterface::isListening() const {
    return impl_->listening;
}

void VoiceInterface::speak(const std::string& text) {
    if (text.empty()) {
        return;
    }

    LOG_INFO(impl_->logger, "Speaking: " + text);
    
    impl_->speaking = true;
    emit speakingStarted(QString::fromStdString(text));

    Event event("voice.speaking.started");
    event.data["text"] = text;
    impl_->eventBus->publish(event);

    impl_->tts->say(QString::fromStdString(text));
}

void VoiceInterface::stopSpeaking() {
    if (impl_->tts && impl_->speaking) {
        impl_->tts->stop();
        impl_->speaking = false;
    }
}

bool VoiceInterface::isSpeaking() const {
    return impl_->speaking;
}

void VoiceInterface::setLanguage(const std::string& language) {
    impl_->language = language;
    LOG_INFO(impl_->logger, "Voice language set to: " + language);
}

void VoiceInterface::setRecognitionEngine(const std::string& engine) {
    impl_->recognitionEngine = engine;
    LOG_INFO(impl_->logger, "Recognition engine set to: " + engine);
}

void VoiceInterface::setSynthesisEngine(const std::string& engine) {
    impl_->synthesisEngine = engine;
    LOG_INFO(impl_->logger, "Synthesis engine set to: " + engine);
}

} // namespace XAI
} // namespace DIREWOLF
