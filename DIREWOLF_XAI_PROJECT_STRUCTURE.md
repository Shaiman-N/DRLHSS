# DIREWOLF XAI System - Complete Project Structure

## Overview

This document defines the complete directory structure, file organization, and configuration for the DIREWOLF XAI production system.

## Root Directory Structure

```
DRLHSS/
├── .github/workflows/          # CI/CD automation
├── .kiro/specs/               # Specification documents
├── cmake/                     # CMake modules
├── config/                    # Configuration files
├── deployment/                # Deployment scripts
├── docs/                      # Documentation
├── external/                  # External dependencies
├── include/xai/              # C++ headers
├── installer/                 # Installation packages
├── python/xai/               # Python modules
├── qml/                      # Qt QML UI files
├── resources/                # Assets and resources
├── scripts/                  # Utility scripts
├── src/xai/                  # C++ source files
├── tests/                    # Test suites
├── tools/                    # Development tools
├── CMakeLists.txt            # Root CMake config
├── conanfile.txt             # Conan dependencies
├── vcpkg.json                # vcpkg dependencies
└── README.md                 # Project documentation
```

## Detailed Structure

### Source Code Organization

```
src/xai/
├── core/
│   ├── Application.cpp           # Main application
│   ├── PluginManager.cpp        # Plugin system
│   ├── EventBus.cpp             # Event handling
│   ├── ConfigManager.cpp        # Configuration
│   └── Logger.cpp               # Logging system
├── voice/
│   ├── SpeechRecognition.cpp    # Speech-to-text
│   ├── TextToSpeech.cpp         # Text-to-speech
│   ├── VoiceBiometrics.cpp      # Voice auth
│   ├── AudioProcessor.cpp       # Audio processing
│   └── VoiceInterface.cpp       # Voice interface
├── nlp/
│   ├── NLUEngine.cpp            # NLU engine
│   ├── IntentClassifier.cpp     # Intent recognition
│   ├── EntityExtractor.cpp      # Entity extraction
│   ├── ConversationManager.cpp  # Conversation handling
│   └── LLMInterface.cpp         # LLM integration
├── gui/
│   ├── MainWindow.cpp           # Main window
│   ├── DashboardWidget.cpp      # Dashboard
│   ├── VoiceWidget.cpp          # Voice interface
│   ├── SecurityWidget.cpp       # Security dashboard
│   └── SystemWidget.cpp         # System monitor
├── security/
│   ├── SecurityIntegration.cpp  # Security integration
│   ├── ThreatAnalyzer.cpp       # Threat analysis
│   ├── ActionExecutor.cpp       # Security actions
│   └── AuditLogger.cpp          # Audit logging
├── system/
│   ├── FileManager.cpp          # File management
│   ├── PerformanceMonitor.cpp   # Performance monitoring
│   ├── SystemInfo.cpp           # System information
│   └── UpdateManager.cpp        # Update system
└── utils/
    ├── Threading.cpp            # Threading utilities
    ├── Crypto.cpp               # Cryptographic utilities
    ├── Network.cpp              # Network utilities
    └── FileUtils.cpp            # File utilities
```

### Header Files

```
include/xai/
├── core/
│   ├── Application.hpp
│   ├── PluginManager.hpp
│   ├── EventBus.hpp
│   ├── ConfigManager.hpp
│   └── Logger.hpp
├── voice/
│   ├── SpeechRecognition.hpp
│   ├── TextToSpeech.hpp
│   ├── VoiceBiometrics.hpp
│   ├── AudioProcessor.hpp
│   └── VoiceInterface.hpp
├── nlp/
│   ├── NLUEngine.hpp
│   ├── IntentClassifier.hpp
│   ├── EntityExtractor.hpp
│   ├── ConversationManager.hpp
│   └── LLMInterface.hpp
├── gui/
│   ├── MainWindow.hpp
│   ├── DashboardWidget.hpp
│   ├── VoiceWidget.hpp
│   ├── SecurityWidget.hpp
│   └── SystemWidget.hpp
├── security/
│   ├── SecurityIntegration.hpp
│   ├── ThreatAnalyzer.hpp
│   ├── ActionExecutor.hpp
│   └── AuditLogger.hpp
├── system/
│   ├── FileManager.hpp
│   ├── PerformanceMonitor.hpp
│   ├── SystemInfo.hpp
│   └── UpdateManager.hpp
└── utils/
    ├── Threading.hpp
    ├── Crypto.hpp
    ├── Network.hpp
    └── FileUtils.hpp
```

### Python Modules

```
python/xai/
├── __init__.py
├── nlp/
│   ├── __init__.py
│   ├── intent_classifier.py
│   ├── entity_extractor.py
│   ├── conversation_manager.py
│   ├── llm_interface.py
│   └── training/
│       ├── train_intent.py
│       ├── train_entity.py
│       └── data_preparation.py
├── voice/
│   ├── __init__.py
│   ├── speech_recognition.py
│   ├── text_to_speech.py
│   ├── voice_biometrics.py
│   └── audio_processing.py
├── ml/
│   ├── __init__.py
│   ├── models/
│   │   ├── intent_model.py
│   │   ├── entity_model.py
│   │   └── voice_model.py
│   └── utils/
│       ├── data_loader.py
│       ├── preprocessing.py
│       └── evaluation.py
└── utils/
    ├── __init__.py
    ├── config.py
    ├── logging.py
    └── helpers.py
```

### QML UI Files

```
qml/
├── main.qml
├── components/
│   ├── VoiceInterface.qml
│   ├── SecurityDashboard.qml
│   ├── SystemMonitor.qml
│   ├── ChatWindow.qml
│   ├── SettingsPanel.qml
│   └── common/
│       ├── Button.qml
│       ├── Card.qml
│       ├── Chart.qml
│       └── Theme.qml
├── pages/
│   ├── Dashboard.qml
│   ├── Security.qml
│   ├── System.qml
│   ├── Settings.qml
│   └── About.qml
└── styles/
    ├── DarkTheme.qml
    ├── LightTheme.qml
    └── Colors.qml
```

### Test Structure

```
tests/
├── unit/
│   ├── core/
│   │   ├── test_application.cpp
│   │   ├── test_plugin_manager.cpp
│   │   └── test_event_bus.cpp
│   ├── voice/
│   │   ├── test_speech_recognition.cpp
│   │   ├── test_tts.cpp
│   │   └── test_voice_biometrics.cpp
│   ├── nlp/
│   │   ├── test_nlu_engine.cpp
│   │   ├── test_intent_classifier.cpp
│   │   └── test_entity_extractor.cpp
│   └── python/
│       ├── test_intent_classifier.py
│       ├── test_entity_extractor.py
│       └── test_voice_processing.py
├── integration/
│   ├── test_voice_to_action.cpp
│   ├── test_security_integration.cpp
│   └── test_gui_integration.cpp
├── performance/
│   ├── benchmark_speech_recognition.cpp
│   ├── benchmark_nlp.cpp
│   └── benchmark_gui.cpp
└── data/
    ├── test_audio/
    ├── test_models/
    └── test_configs/
```

## Configuration Files

### Root CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
project(DIREWOLF_XAI VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(Qt6 REQUIRED COMPONENTS Core Widgets Quick Multimedia)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

include_directories(include)
include_directories(external)

add_subdirectory(src)
add_subdirectory(python)
add_subdirectory(tests)

enable_testing()
```

### Python Requirements

```txt
# requirements.txt
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
pandas>=1.5.0
torch>=2.0.0
transformers>=4.25.0
onnx>=1.14.0
onnxruntime>=1.14.0
librosa>=0.10.0
soundfile>=0.12.0
pyaudio>=0.2.11
openai-whisper>=20230314
speechrecognition>=3.10.0
pyttsx3>=2.90
azure-cognitiveservices-speech>=1.25.0
requests>=2.28.0
pyyaml>=6.0
click>=8.1.0
tqdm>=4.64.0
pytest>=7.2.0
pytest-cov>=4.0.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.0.0
```

### vcpkg Configuration

```json
{
  "name": "direwolf-xai",
  "version": "1.0.0",
  "description": "DIREWOLF XAI System",
  "dependencies": [
    "qt6-base",
    "qt6-declarative",
    "qt6-multimedia",
    "qt6-speech",
    "pybind11",
    "nlohmann-json",
    "spdlog",
    "catch2",
    "openssl",
    "sqlite3",
    "curl",
    "boost-system",
    "boost-filesystem",
    "boost-thread"
  ],
  "builtin-baseline": "2023.12.12"
}
```

### CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  build-and-test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [windows-latest, ubuntu-latest, macos-latest]
        build-type: [Debug, Release]

    steps:
    - uses: actions/checkout@v4
      with:
        submodules: recursive

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install Qt
      uses: jurplel/install-qt-action@v3
      with:
        version: '6.5.0'
        modules: 'qtmultimedia qtquick3d'

    - name: Install vcpkg
      uses: lukka/run-vcpkg@v11
      with:
        vcpkgGitCommitId: '2023.12.12'

    - name: Configure CMake
      run: |
        cmake -B build -DCMAKE_BUILD_TYPE=${{ matrix.build-type }} \
              -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

    - name: Build
      run: cmake --build build --config ${{ matrix.build-type }} --parallel

    - name: Test
      run: |
        cd build
        ctest --output-on-failure --parallel

    - name: Upload Coverage
      if: matrix.os == 'ubuntu-latest' && matrix.build-type == 'Debug'
      uses: codecov/codecov-action@v3
      with:
        file: ./build/coverage.xml
```

## Development Workflow

### Environment Setup

```bash
# Clone repository
git clone https://github.com/your-org/direwolf-xai.git
cd direwolf-xai

# Setup development environment
python tools/development/setup_dev_env.py

# Install dependencies
python scripts/install_dependencies.py

# Setup Python environment
python scripts/setup_python_env.py

# Download ML models
python scripts/download_models.py
```

### Build Process

```bash
# Configure build
cmake -B build -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake

# Build project
cmake --build build --parallel

# Run tests
cd build && ctest --parallel

# Run application
./build/src/direwolf-xai
```

### Development Tasks

```bash
# Format code
cmake --build build --target format

# Generate documentation
cmake --build build --target docs

# Run performance benchmarks
python tools/testing/performance_profiler.py

# Check dependencies
python tools/development/dependency_checker.py
```

## Integration Points

### C++/Python Bridge

```cpp
// Python module binding
PYBIND11_MODULE(direwolf_xai, m) {
    m.doc() = "DIREWOLF XAI Python bindings";
    
    py::class_<VoiceInterface>(m, "VoiceInterface")
        .def(py::init<>())
        .def("start_listening", &VoiceInterface::startListening)
        .def("stop_listening", &VoiceInterface::stopListening)
        .def("process_audio", &VoiceInterface::processAudio);
    
    py::class_<NLPInterface>(m, "NLPInterface")
        .def(py::init<>())
        .def("process_text", &NLPInterface::processText)
        .def("get_intent", &NLPInterface::getIntent)
        .def("extract_entities", &NLPInterface::extractEntities);
}
```

### Qt/QML Integration

```cpp
// Register C++ types with QML
void registerQmlTypes() {
    qmlRegisterType<VoiceInterface>("DIREWOLF.Voice", 1, 0, "VoiceInterface");
    qmlRegisterType<SecurityDashboard>("DIREWOLF.Security", 1, 0, "SecurityDashboard");
    qmlRegisterType<SystemMonitor>("DIREWOLF.System", 1, 0, "SystemMonitor");
}

// Main application setup
int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    
    registerQmlTypes();
    
    QQmlApplicationEngine engine;
    engine.load(QUrl(QStringLiteral("qrc:/qml/main.qml")));
    
    return app.exec();
}
```

This comprehensive project structure provides the foundation for a production-grade DIREWOLF XAI system with proper separation of concerns, comprehensive testing, and professional development workflows.
