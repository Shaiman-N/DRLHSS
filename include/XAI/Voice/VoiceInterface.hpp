#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <QObject>
#include <QAudioFormat>

namespace DIREWOLF {
namespace XAI {

class EventBus;
class Logger;

/**
 * @brief Voice recognition result
 */
struct RecognitionResult {
    std::string text;
    float confidence;
    int latencyMs;
    bool success;
};

/**
 * @brief Voice interface managing speech recognition and synthesis
 */
class VoiceInterface : public QObject {
    Q_OBJECT

public:
    explicit VoiceInterface(EventBus* eventBus, Logger* logger, QObject* parent = nullptr);
    ~VoiceInterface() override;

    // Initialization
    bool initialize();
    void shutdown();

    // Speech recognition
    void startListening();
    void stopListening();
    bool isListening() const;
    
    // Text-to-speech
    void speak(const std::string& text);
    void stopSpeaking();
    bool isSpeaking() const;

    // Configuration
    void setLanguage(const std::string& language);
    void setRecognitionEngine(const std::string& engine);
    void setSynthesisEngine(const std::string& engine);

signals:
    void speechRecognized(const QString& text, float confidence);
    void listeningStarted();
    void listeningStopped();
    void speakingStarted(const QString& text);
    void speakingFinished();
    void errorOccurred(const QString& error);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace XAI
} // namespace DIREWOLF
