# DIREWOLF XAI System

## Overview

The DIREWOLF XAI (Explainable AI) System is an intelligent security assistant that provides natural language interaction, voice commands, and automated security management through conversational AI.

## Current Status

### ✅ Phase 1: Foundation - COMPLETE
### ✅ Phase 2: Voice Interface - COMPLETE

Core application framework with plugin system, event bus, configuration management, logging, and complete voice interface.

**Build & Run**:
```bash
# Windows
build_xai.bat
cd build_xai\Release
direwolf_xai_voice.exe  # Phase 2 with voice
# or direwolf_xai.exe    # Phase 1 only

# Linux/macOS
./build_xai.sh
cd build_xai
./direwolf_xai_voice    # Phase 2 with voice
# or ./direwolf_xai      # Phase 1 only
```

## Features Implemented

### Core Components (Phase 1)
- ✅ **XAIApplication** - Main application framework
- ✅ **PluginManager** - Dynamic plugin loading
- ✅ **EventBus** - Publish-subscribe events
- ✅ **ConfigManager** - JSON configuration
- ✅ **Logger** - Multi-level logging

### Voice Components (Phase 2)
- ✅ **VoiceInterface** - Speech recognition & TTS
- ✅ **VoiceProcessor** - Python speech processing
- ✅ **VoiceBiometrics** - Voice authentication
- ✅ **Qt TextToSpeech** - Audio synthesis

### Capabilities
- Clean startup and shutdown
- Event-driven architecture
- Configuration persistence
- Plugin extensibility
- Comprehensive logging

## Quick Start

### Prerequisites
- Qt 6.5+
- CMake 3.20+
- C++20 compiler
- Windows/Linux/macOS

### Build
```bash
# Clone and build
git clone <repository>
cd DRLHSS
build_xai.bat  # or ./build_xai.sh
```

### Run
```bash
cd build_xai/Release  # or build_xai/
./direwolf_xai
```

## Documentation

| Document | Description |
|----------|-------------|
| [Phase 1 Complete](DIREWOLF_XAI_PHASE1_COMPLETE.md) | Full Phase 1 documentation |
| [Quick Start](XAI_PHASE1_QUICK_START.md) | Quick start guide |
| [Summary](XAI_PHASE1_SUMMARY.md) | Phase 1 summary |
| [Complete Guide](DIREWOLF_XAI_COMPLETE_GUIDE.md) | Full implementation guide |
| [Roadmap](DIREWOLF_XAI_PRODUCTION_ROADMAP.md) | 16-week roadmap |
| [Project Structure](DIREWOLF_XAI_PROJECT_STRUCTURE.md) | Project organization |

## Architecture

```
DIREWOLF XAI
├── Core Framework
│   ├── Application Management
│   ├── Plugin System
│   ├── Event Bus
│   ├── Configuration
│   └── Logging
├── Voice Interface (Phase 2)
│   ├── Speech Recognition
│   ├── Text-to-Speech
│   └── Voice Biometrics
├── NLP Engine (Phase 3)
│   ├── Intent Classification
│   ├── Entity Extraction
│   └── LLM Integration
├── GUI Dashboard (Phase 4)
│   ├── Qt/QML Interface
│   ├── Real-time Monitoring
│   └── Voice Visualization
└── System Integration (Phase 5)
    ├── Security Systems
    ├── File Management
    └── Performance Monitoring
```

## API Examples

### Event Bus
```cpp
auto* eventBus = app.eventBus();

// Subscribe
int id = eventBus->subscribe("event.type", [](const Event& e) {
    // Handle event
});

// Publish
Event event("event.type");
eventBus->publish(event);
```

### Configuration
```cpp
auto* config = app.configManager();

// Set/Get values
config->setValue("key", value);
auto value = config->getValue<Type>("key", default);
```

### Logging
```cpp
auto* logger = app.logger();

logger->info("Message");
logger->error("Error");

// Or use macros
LOG_INFO(logger, "Message");
```

## Development Roadmap

### Phase 1: Foundation ✅ (Weeks 1-2)
- Core application framework
- Plugin system
- Event bus
- Configuration management
- Logging system

### Phase 2: Voice Interface ✅ (Weeks 3-5)
- Speech recognition (Whisper integration ready)
- Text-to-speech (Qt TextToSpeech)
- Voice biometrics (Python implementation)
- Audio processing (Noise reduction)

### Phase 3: NLP Engine (Weeks 6-8)
- Intent classification
- Entity extraction
- LLM integration
- Conversation management

### Phase 4: GUI Dashboard (Weeks 9-11)
- Qt/QML interface
- Real-time monitoring
- Voice visualization
- Settings interface

### Phase 5: System Integration (Weeks 12-14)
- Security system integration
- File management
- Performance monitoring
- Automation

### Phase 6: Testing & Deployment (Weeks 15-16)
- Comprehensive testing
- Performance optimization
- Security audit
- Production deployment

## Contributing

### Code Style
- C++20 standard
- Modern C++ practices
- Doxygen comments
- Thread-safe code

### Building
```bash
mkdir build_xai
cd build_xai
cmake -DCMAKE_BUILD_TYPE=Release -f ../CMakeLists_xai.txt ..
cmake --build . --parallel
```

### Testing
```bash
cd build_xai
./direwolf_xai
```

## Requirements

### Development
- Qt 6.5+
- CMake 3.20+
- C++20 compiler (MSVC 2019+, GCC 10+, Clang 10+)
- Git

### Runtime
- Windows 10/11, Linux, or macOS
- 8GB RAM minimum
- 10GB disk space
- Microphone (for voice features)

## License

Copyright © 2024 DIREWOLF Project. All rights reserved.

## Support

- **Documentation**: See docs/ directory
- **Issues**: GitHub Issues
- **Questions**: GitHub Discussions

## Status

**Current Phase**: Phase 2 ✅ COMPLETE
**Next Phase**: Phase 3 - NLP Engine
**Overall Progress**: 33% (2/6 phases)

---

**Last Updated**: November 28, 2024
**Version**: 1.0.0-phase1
