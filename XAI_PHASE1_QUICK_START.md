# DIREWOLF XAI Phase 1 - Quick Start Guide

## Build & Run in 3 Steps

### Step 1: Build
```bash
# Windows
build_xai.bat

# Linux/macOS
chmod +x build_xai.sh
./build_xai.sh
```

### Step 2: Run
```bash
# Windows
cd build_xai\Release
direwolf_xai.exe

# Linux/macOS
cd build_xai
./direwolf_xai
```

### Step 3: Verify
You should see:
```
✓ Application initialized successfully
✓ Event Bus working
✓ Configuration Manager working
✓ Logger working
✓ Plugin Manager working
```

## What Was Built

### Core Components
1. **XAIApplication** - Main application framework
2. **PluginManager** - Dynamic plugin loading
3. **EventBus** - Publish-subscribe events
4. **ConfigManager** - JSON configuration
5. **Logger** - Multi-level logging

### Key Features
- ✅ Clean startup/shutdown
- ✅ Event propagation
- ✅ Configuration persistence
- ✅ Plugin system ready
- ✅ Comprehensive logging

## Quick API Examples

### Using the Event Bus
```cpp
auto* eventBus = app.eventBus();

// Subscribe to events
int id = eventBus->subscribe("my.event", [](const Event& e) {
    std::cout << "Event: " << e.type << std::endl;
});

// Publish events
Event event("my.event");
event.data["message"] = std::string("Hello!");
eventBus->publish(event);

// Unsubscribe
eventBus->unsubscribe("my.event", id);
```

### Using Configuration
```cpp
auto* config = app.configManager();

// Set values
config->setValue("app.timeout", 30);
config->setValue("app.name", std::string("MyApp"));

// Get values
int timeout = config->getValue<int>("app.timeout", 10);
std::string name = config->getValue<std::string>("app.name", "Default");

// Save configuration
config->saveConfig("config.json");
```

### Using Logger
```cpp
auto* logger = app.logger();

logger->info("Application started");
logger->warning("Low memory");
logger->error("Connection failed");

// Or use macros
LOG_INFO(logger, "This is easier");
LOG_ERROR(logger, "Something went wrong");
```

### Creating a Plugin
```cpp
class MyPlugin : public IPlugin {
public:
    bool initialize() override {
        // Initialize plugin
        return true;
    }
    
    void shutdown() override {
        // Cleanup
    }
    
    std::string getName() const override {
        return "MyPlugin";
    }
    
    std::string getVersion() const override {
        return "1.0.0";
    }
    
    std::string getDescription() const override {
        return "My custom plugin";
    }
};

// Export function
extern "C" IPlugin* createPlugin() {
    return new MyPlugin();
}
```

## File Locations

### Source Code
- Headers: `include/XAI/Core/`
- Implementation: `src/XAI/Core/`
- Main: `src/XAI/xai_main.cpp`

### Build Files
- CMake: `CMakeLists_xai.txt`
- Build scripts: `build_xai.bat`, `build_xai.sh`
- Build output: `build_xai/`

### Runtime Files
- Config: `%APPDATA%/DIREWOLF/config.json` (Windows)
- Logs: `%APPDATA%/DIREWOLF/direwolf_xai.log` (Windows)
- Plugins: `%APPDATA%/DIREWOLF/plugins/` (Windows)

## Troubleshooting

### Qt Not Found
```bash
# Set Qt path
export Qt6_DIR=/path/to/Qt/6.5.3/gcc_64
```

### Build Fails
```bash
# Check CMake version
cmake --version  # Should be 3.20+

# Check compiler
g++ --version    # Should support C++20
```

### Application Crashes
```bash
# Check logs
cat ~/.local/share/DIREWOLF/direwolf_xai.log
```

## Next Steps

Phase 1 is complete! Ready for:
- **Phase 2**: Voice Interface (Speech recognition, TTS)
- **Phase 3**: NLP Engine (Intent classification, LLM)
- **Phase 4**: GUI Dashboard (Qt/QML interface)

## Documentation

- Full details: `DIREWOLF_XAI_PHASE1_COMPLETE.md`
- Complete guide: `DIREWOLF_XAI_COMPLETE_GUIDE.md`
- Project structure: `DIREWOLF_XAI_PROJECT_STRUCTURE.md`
- Roadmap: `DIREWOLF_XAI_PRODUCTION_ROADMAP.md`

## Support

For issues or questions:
1. Check `DIREWOLF_XAI_PHASE1_COMPLETE.md`
2. Review code comments in headers
3. Examine test application in `xai_main.cpp`

---

**Phase 1 Status**: ✅ COMPLETE
**Ready for Phase 2**: YES
