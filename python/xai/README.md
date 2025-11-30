# DIREWOLF XAI Python Components

## Phase 1 Implementation - Complete ✅

This directory contains the Python components for DIREWOLF's Explainable AI system.

---

## 📦 Components

### 1. LLM Engine (`llm_engine.py`)
Dynamic conversation generation using Large Language Models.

**Features**:
- Wolf's personality implementation
- Context-aware response generation
- Hybrid local/cloud support
- Urgency-based tone adaptation

**Quick Start**:
```python
from llm_engine import LLMEngine, SystemState, ConversationContext, UrgencyLevel

# Configure
config = {
    'mode': 'hybrid',
    'cloud_provider': 'openai',
    'cloud_api_key': 'your-key'
}

# Initialize
engine = LLMEngine(config)

# Create context
system_state = SystemState(
    threats_today=12,
    active_alerts=2,
    health_status="HEALTHY",
    drl_confidence=0.94,
    recent_events=[],
    component_status={'AV': 'RUNNING', 'NIDPS': 'RUNNING'}
)

context = ConversationContext(
    user_id='alpha_001',
    recent_exchanges=[],
    user_preferences={},
    ongoing_topics=[]
)

# Generate response
response = engine.generate_response(
    user_input="What's the security status?",
    context=context,
    system_state=system_state,
    urgency=UrgencyLevel.ROUTINE
)

print(f"Wolf: {response}")
```

---

### 2. Voice Interface (`voice_interface.py`)
Text-to-Speech, Speech-to-Text, and wake word detection.

**Features**:
- Multi-provider TTS (Azure, Google, Coqui)
- Multi-provider STT (Whisper, Azure, Google)
- Wake word detection (Porcupine)
- Urgency-based voice modulation
- Interrupt capability

**Quick Start**:
```python
from voice_interface import VoiceInterface, UrgencyLevel

# Configure
config = {
    'tts_provider': 'azure',
    'stt_provider': 'whisper',
    'wake_word_provider': 'porcupine',
    'azure_speech_key': 'your-key',
    'azure_speech_region': 'eastus',
    'voice_name': 'en-US-GuyNeural',
    'wake_word': 'hey wolf'
}

# Initialize
voice = VoiceInterface(config)
voice.start()

# Speak with different urgency levels
voice.speak("Alpha, your network is secure.", UrgencyLevel.ROUTINE)
voice.speak("Alpha, suspicious activity detected.", UrgencyLevel.ELEVATED)
voice.speak("Alpha, critical threat detected!", UrgencyLevel.CRITICAL)
voice.speak("Alpha, EMERGENCY! Active breach!", UrgencyLevel.EMERGENCY, interrupt=True)

# Listen for commands
def on_speech(text: str):
    print(f"Alpha said: {text}")

voice.start_listening(on_speech)

# Wake word detection
def on_wake_word():
    print("Wake word detected!")
    voice.speak("Yes, Alpha?")

voice.on_wake_word = on_wake_word
```

**Voice Modulation**:
| Urgency | Rate | Pitch | Volume | Use Case |
|---------|------|-------|--------|----------|
| Routine | 1.0x | 0% | 100% | Normal status updates |
| Elevated | 1.1x | +5% | 110% | Suspicious activity |
| Critical | 1.2x | +10% | 120% | Confirmed threats |
| Emergency | 1.3x | +15% | 130% | Active breaches |

---

### 3. Conversation Manager (`conversation_manager.py`)
Tracks conversations, learns from decisions, adapts to Alpha's style.

**Features**:
- SQLite-backed conversation history
- User profile management
- Decision history tracking
- Communication style adaptation
- Pattern recognition

**Quick Start**:
```python
from conversation_manager import ConversationManager

# Configure
config = {
    'database_path': 'direwolf_conversations.db',
    'max_history_days': 90,
    'learning_enabled': True
}

# Initialize
conv_manager = ConversationManager(config)

# Start conversation
session_id = conv_manager.start_conversation("alpha_001")

# Record exchange
conv_manager.add_exchange(
    user_id="alpha_001",
    user_input="What's the security status?",
    wolf_response="Alpha, your network is secure. I've blocked 3 threats today.",
    context="routine_check",
    urgency_level="routine",
    response_time_ms=1200
)

# Record Alpha's decision
conv_manager.record_alpha_decision(
    user_id="alpha_001",
    threat_type="malware",
    wolf_recommendation="quarantine_file",
    alpha_decision="approved",
    confidence_level=0.94
)

# Get context for LLM
context = conv_manager.get_conversation_context("alpha_001")
print(f"Context: {context}")

# Analyze communication style
analysis = conv_manager.analyze_communication_style("alpha_001")
print(f"Alpha prefers brief responses: {analysis['prefers_brief_responses']}")
print(f"Technical expertise: {analysis['technical_expertise']}")

# Get decision patterns
patterns = conv_manager.get_decision_patterns("alpha_001")
for pattern in patterns:
    print(f"Pattern: {pattern}")
```

**Learning Features**:
- Analyzes every 10 exchanges
- Identifies technical expertise level
- Detects brief vs detailed preference
- Recognizes decision patterns (3+ rejections)
- Adapts Wolf's responses

---

### 4. Development Auto-Update (`dev_auto_update.py`)
Automatically rebuilds and hot-reloads code changes during development.

**Features**:
- File watcher for source changes
- Automatic C++ rebuild
- Hot reload for Python modules
- Build cooldown management

**Quick Start**:
```python
from dev_auto_update import DevAutoUpdate

# Configure
config = {
    'enabled': True,
    'project_root': '.',
    'watch_paths': ['./src', './include', './python'],
    'build_dir': 'build',
    'file_extensions': ['.cpp', '.hpp', '.py', '.qml'],
    'ignore_patterns': ['__pycache__', '.git', 'build'],
    'scan_interval': 2.0,
    'build_cooldown': 5.0,
    'cmake_args': []
}

# Initialize
auto_update = DevAutoUpdate(config)

# Callback for update completion
def on_update(changed_files, file_types):
    print(f"Update completed for: {file_types}")
    print(f"Changed files: {changed_files}")

# Start
auto_update.start(on_update_complete=on_update)

# Keep running
try:
    print("Auto-update system running. Press Ctrl+C to stop.")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    auto_update.stop()
```

**How It Works**:
1. Monitors source directories for changes
2. Calculates SHA-256 hash of each file
3. Detects changes and triggers appropriate action:
   - C++ files → CMake rebuild
   - Python files → Hot reload modules
   - QML files → Qt hot reload
4. Notifies on completion

---

## 🔧 Installation

### Required Python Packages

```bash
# Core dependencies
pip install sqlite3  # Built-in

# LLM Engine (choose one or both)
pip install openai  # For OpenAI API
pip install anthropic  # For Anthropic API

# Voice Interface - TTS
pip install azure-cognitiveservices-speech  # Azure TTS/STT
pip install google-cloud-texttospeech  # Google TTS
pip install TTS  # Coqui TTS (local)

# Voice Interface - STT
pip install openai-whisper  # Whisper STT (local)
pip install google-cloud-speech  # Google STT

# Voice Interface - Wake Word
pip install pvporcupine  # Porcupine wake word

# Development Auto-Update
# No additional packages needed (uses subprocess)
```

### Optional Dependencies

```bash
# For audio playback/recording
pip install pyaudio
pip install sounddevice

# For local LLM support
pip install llama-cpp-python
pip install transformers
```

---

## 🎯 Integration Example

Complete integration of all Phase 1 components:

```python
import time
from llm_engine import LLMEngine, SystemState, ConversationContext, UrgencyLevel
from voice_interface import VoiceInterface
from conversation_manager import ConversationManager
from dev_auto_update import DevAutoUpdate

# Configuration
config = {
    'llm': {
        'mode': 'hybrid',
        'cloud_provider': 'openai',
        'cloud_api_key': 'your-key'
    },
    'voice': {
        'tts_provider': 'azure',
        'stt_provider': 'whisper',
        'wake_word_provider': 'porcupine',
        'azure_speech_key': 'your-key',
        'voice_name': 'en-US-GuyNeural',
        'wake_word': 'hey wolf'
    },
    'conversation': {
        'database_path': 'direwolf_conversations.db',
        'learning_enabled': True
    },
    'dev_update': {
        'enabled': True,
        'project_root': '.',
        'watch_paths': ['./src', './include', './python']
    }
}

# Initialize components
llm_engine = LLMEngine(config['llm'])
voice = VoiceInterface(config['voice'])
conv_manager = ConversationManager(config['conversation'])
auto_update = DevAutoUpdate(config['dev_update'])

# Start services
voice.start()
auto_update.start()
session_id = conv_manager.start_conversation("alpha_001")

# Handle speech input
def on_speech(text: str):
    print(f"Alpha: {text}")
    
    # Get conversation context
    context = conv_manager.get_conversation_context("alpha_001")
    
    # Get system state (would come from DRLHSS)
    system_state = SystemState(
        threats_today=12,
        active_alerts=2,
        health_status="HEALTHY",
        drl_confidence=0.94,
        recent_events=[],
        component_status={'AV': 'RUNNING', 'NIDPS': 'RUNNING'}
    )
    
    # Generate response
    response = llm_engine.generate_response(
        user_input=text,
        context=context,
        system_state=system_state,
        urgency=UrgencyLevel.ROUTINE
    )
    
    # Speak response
    voice.speak(response, UrgencyLevel.ROUTINE)
    
    # Record exchange
    conv_manager.add_exchange(
        user_id="alpha_001",
        user_input=text,
        wolf_response=response,
        context="conversation",
        urgency_level="routine"
    )

# Handle wake word
def on_wake_word():
    voice.speak("Yes, Alpha?", UrgencyLevel.ROUTINE)

# Connect callbacks
voice.start_listening(on_speech)
voice.on_wake_word = on_wake_word

# Keep running
try:
    print("DIREWOLF is active. Say 'hey wolf' to start.")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    voice.stop()
    auto_update.stop()
    print("DIREWOLF stopped.")
```

---

## 🧪 Testing

### Unit Tests

```python
# Test LLM Engine
def test_llm_engine():
    engine = LLMEngine({'mode': 'local'})
    response = engine.generate_response(
        user_input="Test",
        context=mock_context,
        system_state=mock_state,
        urgency=UrgencyLevel.ROUTINE
    )
    assert response is not None
    assert "Alpha" in response

# Test Voice Interface
def test_voice_interface():
    voice = VoiceInterface({'tts_provider': 'coqui'})
    voice.start()
    voice.speak("Test", UrgencyLevel.ROUTINE)
    voice.stop()

# Test Conversation Manager
def test_conversation_manager():
    conv = ConversationManager({'database_path': ':memory:'})
    session = conv.start_conversation("test_user")
    assert session is not None
    
    conv.add_exchange(
        user_id="test_user",
        user_input="Test",
        wolf_response="Response"
    )
    
    context = conv.get_conversation_context("test_user")
    assert len(context['recent_exchanges']) == 1

# Test Auto-Update
def test_auto_update():
    auto = DevAutoUpdate({'enabled': True, 'project_root': '.'})
    auto.start()
    time.sleep(5)
    auto.stop()
```

---

## 📊 Performance

### Typical Latencies
- **LLM Response**: 1-3 seconds (cloud), 2-5 seconds (local)
- **Voice Synthesis**: 1-2 seconds per sentence
- **Voice Recognition**: 1-2 seconds per utterance
- **Wake Word Detection**: < 100ms
- **File Change Detection**: 2 second scan interval
- **Build Trigger**: < 1 second
- **Hot Reload**: < 500ms

### Resource Usage
- **Memory**: ~200-500 MB (without local LLM)
- **Memory**: ~2-4 GB (with local LLM)
- **CPU**: < 5% idle, 20-40% during speech/LLM
- **Disk**: ~100 MB for conversation database

---

## 🔒 Security Considerations

### API Keys
- Store API keys in environment variables
- Never commit keys to version control
- Use key rotation for production

### Database
- Conversation database contains sensitive data
- Encrypt database file in production
- Implement access controls

### Voice Data
- Audio data should be processed locally when possible
- Clear audio buffers after processing
- Implement voice authentication for production

---

## 🐛 Troubleshooting

### LLM Engine Issues
```python
# Check if LLM is responding
response = engine._fallback_response()
# Should return: "Alpha, I'm experiencing difficulty..."
```

### Voice Interface Issues
```python
# Check TTS provider
if not voice.tts_engine:
    print("TTS not initialized")
    
# Check STT provider
if not voice.stt_engine:
    print("STT not initialized")
```

### Conversation Manager Issues
```python
# Check database connection
import sqlite3
conn = sqlite3.connect('direwolf_conversations.db')
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM conversation_exchanges")
print(f"Total exchanges: {cursor.fetchone()[0]}")
```

### Auto-Update Issues
```python
# Check file watcher
print(f"Watching: {auto_update.file_watcher.watch_paths}")
print(f"Tracking: {len(auto_update.file_watcher.file_hashes)} files")
```

---

## 📚 Additional Resources

- **Phase 1 Complete**: `../DIREWOLF_PHASE1_COMPLETE.md`
- **Implementation Phases**: `../DIREWOLF_IMPLEMENTATION_PHASES.md`
- **Specifications**: `../.kiro/specs/direwolf-xai-system/`
- **Quick Start**: `../docs/DIREWOLF_QUICKSTART.md`

---

## 🐺 The Pack Protects. The Wolf Explains. Alpha Commands.

**Phase 1 Status**: ✅ COMPLETE  
**Next Phase**: Phase 2 - Data Integration & Core Engine

---

*Last Updated: November 27, 2025*
