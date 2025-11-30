# DIREWOLF XAI Phase 2 - Deployment Guide

## Quick Deploy (5 Minutes)

### Step 1: Build
```bash
# Windows
build_xai.bat

# Linux/macOS
chmod +x build_xai.sh
./build_xai.sh
```

### Step 2: Run Voice Test
```bash
# Windows
cd build_xai\Release
direwolf_xai_voice.exe

# Linux/macOS
cd build_xai
./direwolf_xai_voice
```

### Step 3: Verify
You should see:
```
✓ Application initialized successfully
✓ Text-to-Speech working
✓ Speech Recognition working
✓ Event Bus integration working
✓ Configuration working

Phase 2 Voice Interface - All Tests Passed! ✓
```

## What's New in Phase 2

### Components Added
1. **VoiceInterface** - C++ voice control
2. **VoiceProcessor** - Python speech processing
3. **VoiceBiometrics** - Voice authentication
4. **Qt TextToSpeech** - Audio output

### Features
- ✅ Speech recognition framework
- ✅ Text-to-speech synthesis
- ✅ Voice event system
- ✅ Configuration management
- ✅ Python integration ready

## Files Created (8 new files)

```
include/XAI/Voice/VoiceInterface.hpp
src/XAI/Voice/VoiceInterface.cpp
python/xai/voice_processor.py
python/xai/requirements_voice.txt
src/XAI/xai_voice_test.cpp
DIREWOLF_XAI_PHASE2_COMPLETE.md
XAI_PHASE2_DEPLOY.md (this file)
```

## Production Deployment

### Prerequisites
- Qt 6.5+ with TextToSpeech module
- Python 3.11+ with numpy
- (Optional) OpenAI Whisper for full speech recognition

### Install Python Dependencies
```bash
cd python/xai
pip install numpy

# Optional: Full Whisper support
pip install openai-whisper torch
```

### Build for Production
```bash
cmake -B build_xai -DCMAKE_BUILD_TYPE=Release
cmake --build build_xai --config Release
cmake --install build_xai
```

### Run in Production
```bash
# Start voice-enabled application
direwolf_xai_voice

# Or use the base application
direwolf_xai
```

## API Usage

### C++ Example
```cpp
#include "XAI/Voice/VoiceInterface.hpp"

auto* voice = app.voiceInterface();

// Speak
voice->speak("System ready");

// Listen
voice->startListening();

// Handle recognition
connect(voice, &VoiceInterface::speechRecognized,
        [](const QString& text, float conf) {
    // Process recognized speech
});
```

### Python Example
```python
from xai.voice_processor import get_voice_processor

processor = get_voice_processor()
result = processor.recognize_speech(audio_data)
print(result['text'])
```

## Configuration

Edit `config.json`:
```json
{
    "voice": {
        "language": "en-US",
        "recognition_engine": "whisper",
        "synthesis_engine": "qt"
    }
}
```

## Troubleshooting

### No Audio Output
- Check system audio settings
- Verify Qt TextToSpeech installation
- Test with: `voice->speak("test")`

### Build Fails
- Ensure Qt TextToSpeech module installed
- Check CMake finds Qt6::TextToSpeech
- Verify C++20 compiler support

### Python Import Errors
```bash
pip install numpy
# System works without Whisper
```

## Next Steps

Phase 2 is complete! Ready for:
- **Phase 3**: NLP Engine (Intent classification, LLM)
- **Phase 4**: GUI Dashboard (Qt/QML interface)
- **Phase 5**: System Integration (Security, files)

## Support

- Full docs: `DIREWOLF_XAI_PHASE2_COMPLETE.md`
- Phase 1 docs: `DIREWOLF_XAI_PHASE1_COMPLETE.md`
- Complete guide: `DIREWOLF_XAI_COMPLETE_GUIDE.md`

---

**Status**: ✅ READY TO DEPLOY
**Build Time**: < 5 minutes
**Test Time**: < 1 minute
