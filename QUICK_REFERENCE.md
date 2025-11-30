# DRLHSS Quick Reference Card

## 🚀 Quick Start

### Build
```bash
# Linux/macOS
./build_all.sh

# Windows
build_all.bat
```

### Run
```bash
# Integrated system
./build/integrated_system_example

# Tests
./build/test_linux_sandbox      # Linux
./build/test_macos_sandbox      # macOS
.\build\Release\test_windows_sandbox.exe  # Windows
```

---

## 📁 Project Structure

```
DRLHSS/
├── include/
│   ├── Sandbox/              # Sandbox interfaces
│   ├── Detection/            # Detection components
│   ├── DRL/                  # DRL components
│   └── DB/                   # Database components
├── src/
│   ├── Sandbox/              # Sandbox implementations
│   ├── Detection/            # Detection implementations
│   ├── DRL/                  # DRL implementations
│   └── DB/                   # Database implementations
├── tests/                    # Test suites
├── docs/                     # Documentation
├── models/                   # ONNX models
└── config/                   # Configuration files
```

---

## 🔧 Key Components

### 1. Unified Detection Coordinator
```cpp
#include "Detection/UnifiedDetectionCoordinator.hpp"

detection::UnifiedDetectionCoordinator coordinator(
    "models/onnx/mtl_model.onnx", "drlhss.db"
);
coordinator.initialize();
coordinator.start();

// Process packet
auto response = coordinator.processNetworkPacket(packet);
```

### 2. NIDPS Detection Bridge
```cpp
#include "Detection/NIDPSDetectionBridge.hpp"

detection::NIDPSDetectionBridge bridge(drl_orchestrator, "drlhss.db");
bridge.initialize();

// Process packet
auto response = bridge.processPacket(packet);
```

### 3. Cross-Platform Sandboxes
```cpp
#include "Sandbox/SandboxFactory.hpp"

auto sandbox = sandbox::SandboxFactory::createSandbox(
    sandbox::SandboxType::POSITIVE_FP
);

sandbox::SandboxConfig config;
config.memory_limit_mb = 512;
config.cpu_limit_percent = 50;
config.timeout_seconds = 30;

sandbox->initialize(config);
auto result = sandbox->analyzePacket(packet_data);
```

### 4. DRL Orchestrator
```cpp
#include "DRL/DRLOrchestrator.hpp"

drl::DRLOrchestrator orchestrator(
    "models/onnx/mtl_model.onnx", "drlhss.db", 16
);
orchestrator.initialize();

auto response = orchestrator.processWithDetails(telemetry);
```

---

## 📊 Common Operations

### Get Statistics
```cpp
auto stats = coordinator.getStats();
std::cout << "Total: " << stats.total_detections << "\n";
std::cout << "Malicious: " << stats.malicious_detected << "\n";
```

### Export Data
```cpp
// Export detection events
coordinator.exportDetectionEvents("events.csv", 1000);

// Export experiences
orchestrator.exportExperiences("experiences.json", 1000);
```

### Process Different Sources
```cpp
// Network packet
coordinator.processNetworkPacket(packet);

// File
coordinator.processFile("/path/to/file", "hash");

// Behavior
coordinator.processBehavior(telemetry);
```

---

## 🔒 Security Configuration

### Sandbox Limits
```cpp
config.memory_limit_mb = 512;      // Memory limit
config.cpu_limit_percent = 50;     // CPU limit
config.timeout_seconds = 30;       // Timeout
config.allow_network = true;       // Network
config.read_only_filesystem = false;
```

### DRL Thresholds
```cpp
// In code
if (response.confidence > 0.7f && response.is_malicious) {
    // High confidence malicious
}
```

---

## 📈 Performance Tips

1. **Reuse Sandboxes**: Keep initialized, call `reset()` instead of recreating
2. **Batch Processing**: Process multiple items before sandbox analysis
3. **Async Operations**: Use background threads for heavy operations
4. **Database Batching**: Batch database writes
5. **Model Caching**: Keep ONNX model loaded

---

## 🐛 Troubleshooting

### Sandbox Init Fails
```bash
# Linux - Check privileges
sudo ./integrated_system_example
# or
sudo setcap cap_sys_admin+ep ./integrated_system_example

# Windows - Run as Administrator

# macOS - Disable SIP (testing only)
csrutil disable
```

### ONNX Model Not Found
```bash
# Check path
ls -l models/onnx/mtl_model.onnx

# Download if missing
# See docs/DEPLOYMENT_GUIDE.md
```

### Database Errors
```bash
# Check permissions
ls -l drlhss.db

# Check disk space
df -h .

# Repair database
sqlite3 drlhss.db "PRAGMA integrity_check;"
```

---

## 📚 Documentation

- **[Integration Guide](docs/NIDPS_INTEGRATION_GUIDE.md)** - Complete integration
- **[Architecture](docs/CROSS_PLATFORM_SANDBOX_ARCHITECTURE.md)** - System design
- **[Deployment](docs/DEPLOYMENT_GUIDE.md)** - Production deployment
- **[Complete README](COMPLETE_INTEGRATION_README.md)** - Full overview

---

## 🎯 Action Codes

| Code | Action | Description |
|------|--------|-------------|
| 0 | Allow | Permit the artifact |
| 1 | Block | Block the artifact |
| 2 | Quarantine | Isolate for analysis |
| 3 | Deep Scan | Perform detailed analysis |

---

## 🔢 Threat Scores

| Range | Level | Action |
|-------|-------|--------|
| 0-30 | Low | Usually allow |
| 31-60 | Medium | Quarantine/scan |
| 61-80 | High | Block |
| 81-100 | Critical | Block immediately |

---

## 🛠️ Build Targets

```bash
# All targets
cmake --build . --target all

# Specific targets
cmake --build . --target integrated_system_example
cmake --build . --target test_linux_sandbox
cmake --build . --target drl_integration_example
```

---

## 📦 Dependencies

### Required
- CMake 3.15+
- C++17 compiler
- SQLite3
- OpenSSL
- ONNX Runtime 1.16.0+

### Platform-Specific
**Linux**: libseccomp-dev, libpcap-dev  
**Windows**: Windows SDK, Visual Studio 2019+  
**macOS**: Xcode Command Line Tools, Homebrew  

---

## 🔗 Quick Links

- **GitHub**: https://github.com/your-org/drlhss
- **Issues**: https://github.com/your-org/drlhss/issues
- **Docs**: https://docs.drlhss.org
- **Support**: support@drlhss.org

---

## 💡 Tips

1. Start with `integrated_system_example` to see everything working
2. Check logs for detailed information
3. Use statistics API for monitoring
4. Export data regularly for training
5. Keep ONNX model updated

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Status**: Production Ready ✅

