# NIDPS Integration Guide

## Overview

This guide explains how the Network Intrusion Detection and Prevention System (NIDPS) is integrated with the DRL-based malware detection system and cross-platform sandboxes.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Unified Detection Coordinator               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Network    │  │     File     │  │    Behavior      │  │
│  │  Detection   │  │  Detection   │  │   Detection      │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────────┘  │
└─────────┼──────────────────┼──────────────────┼──────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
    ┌─────────────────────────────────────────────────┐
    │          NIDPS Detection Bridge                 │
    │  ┌──────────────────────────────────────────┐  │
    │  │         DRL Orchestrator                 │  │
    │  │  ┌────────────┐  ┌──────────────────┐   │  │
    │  │  │ Inference  │  │  Pattern Learning│   │  │
    │  │  └────────────┘  └──────────────────┘   │  │
    │  └──────────────────────────────────────────┘  │
    │                                                 │
    │  ┌──────────────┐  ┌──────────────────────┐   │
    │  │   Positive   │  │     Negative         │   │
    │  │   Sandbox    │  │     Sandbox          │   │
    │  └──────────────┘  └──────────────────────┘   │
    └─────────────────────────────────────────────────┘
                          │
                          ▼
                ┌──────────────────┐
                │  Database        │
                │  - Telemetry     │
                │  - Experiences   │
                │  - Patterns      │
                └──────────────────┘
```

## Components

### 1. UnifiedDetectionCoordinator

The central coordinator that manages all detection sources:

- **Network Detection**: Processes packets from NIDPS
- **File Detection**: Analyzes files for malware
- **Behavior Detection**: Monitors runtime behavior

**Key Features:**
- Unified statistics across all detection types
- Centralized database persistence
- Event queuing and processing
- Export capabilities for training data

### 2. NIDPSDetectionBridge

Bridges NIDPS packet data with DRL decision-making:

- Converts NIDPS packets to DRL telemetry format
- Coordinates cross-platform sandbox analysis
- Computes rewards for reinforcement learning
- Stores experiences for training

**Key Features:**
- Packet feature extraction
- Hash computation for deduplication
- Sandbox result conversion
- Decision callbacks

### 3. Cross-Platform Sandboxes

Platform-specific sandbox implementations:

#### Linux Sandbox
- **Isolation**: Namespaces (PID, NET, MNT, UTS, IPC)
- **Filesystem**: Overlay filesystem (OverlayFS)
- **Resource Limits**: cgroups v2
- **Security**: seccomp syscall filtering
- **Monitoring**: /proc filesystem, netstat, ps

#### Windows Sandbox
- **Isolation**: Job Objects, AppContainer
- **Filesystem**: Virtual filesystem redirection
- **Registry**: Registry virtualization
- **Resource Limits**: Job Object limits
- **Monitoring**: Windows API (FindFirstFile, RegQueryInfoKey, CreateToolhelp32Snapshot)

#### macOS Sandbox
- **Isolation**: Sandbox Profile Language (SBPL)
- **Filesystem**: Temporary sandbox directory
- **Security**: Code signing verification, TCC restrictions
- **Resource Limits**: setrlimit
- **Monitoring**: lsof, ps, find

## Integration Flow

### Packet Processing Flow

```
1. Packet Captured
   ↓
2. Convert to Telemetry (NIDPSDetectionBridge)
   ↓
3. DRL Inference (DRLOrchestrator)
   ↓
4. Decision: Allow / Block / Quarantine / Deep Scan
   ↓
5. If High Risk → Sandbox Analysis
   ↓
6. Re-evaluate with Sandbox Results
   ↓
7. Store Experience for Learning
   ↓
8. Persist to Database
```

### Sandbox Analysis Flow

```
1. Packet Data → Temporary File
   ↓
2. Initialize Sandbox (Platform-Specific)
   ↓
3. Execute in Isolated Environment
   ↓
4. Monitor Behavior:
   - File system modifications
   - Registry changes (Windows)
   - Network activity
   - Process creation
   - Memory injection
   - API calls
   ↓
5. Calculate Threat Score
   ↓
6. Return SandboxResult
```

## Usage Examples

### Basic Integration

```cpp
#include "Detection/UnifiedDetectionCoordinator.hpp"

// Initialize coordinator
detection::UnifiedDetectionCoordinator coordinator(
    "models/onnx/mtl_model.onnx",
    "drlhss.db"
);

if (!coordinator.initialize()) {
    std::cerr << "Failed to initialize\n";
    return 1;
}

coordinator.start();

// Process network packet
nidps::PacketPtr packet = capturePacket();
auto response = coordinator.processNetworkPacket(packet);

if (response.action == 1) {
    // Block packet
    blockPacket(packet);
} else if (response.action == 2) {
    // Quarantine packet
    quarantinePacket(packet);
}

// Get statistics
auto stats = coordinator.getStats();
std::cout << "Total detections: " << stats.total_detections << "\n";
std::cout << "Malicious detected: " << stats.malicious_detected << "\n";

coordinator.stop();
```

### Direct NIDPS Bridge Usage

```cpp
#include "Detection/NIDPSDetectionBridge.hpp"

// Create DRL orchestrator
auto drl = std::make_shared<drl::DRLOrchestrator>(
    "models/onnx/mtl_model.onnx",
    "drlhss.db"
);
drl->initialize();

// Create NIDPS bridge
detection::NIDPSDetectionBridge bridge(drl, "drlhss.db");
bridge.initialize();

// Set decision callback
bridge.setDecisionCallback([](const nidps::PacketPtr& packet, 
                              int action, float confidence) {
    std::cout << "Packet " << packet->packet_id 
              << " - Action: " << action 
              << " - Confidence: " << confidence << "\n";
});

// Process packet
nidps::PacketPtr packet = createPacket();
auto response = bridge.processPacket(packet);

// Analyze in sandbox
auto sandbox_result = bridge.analyzeSandbox(packet, 
                                           nidps::SandboxType::POSITIVE);
```

### Sandbox Direct Usage

```cpp
#include "Sandbox/SandboxFactory.hpp"

// Create platform-specific sandbox
auto sandbox = sandbox::SandboxFactory::createSandbox(
    sandbox::SandboxType::POSITIVE_FP
);

// Configure
sandbox::SandboxConfig config;
config.memory_limit_mb = 512;
config.cpu_limit_percent = 50;
config.timeout_seconds = 30;

sandbox->initialize(config);

// Analyze packet
std::vector<uint8_t> packet_data = getPacketData();
auto result = sandbox->analyzePacket(packet_data);

std::cout << "Threat score: " << result.threat_score << "\n";
std::cout << "File system modified: " << result.file_system_modified << "\n";
std::cout << "Network activity: " << result.network_activity_detected << "\n";

sandbox->cleanup();
```

## Configuration

### Sandbox Configuration

```cpp
sandbox::SandboxConfig config;

// Resource limits
config.memory_limit_mb = 512;        // Memory limit in MB
config.cpu_limit_percent = 50;       // CPU usage limit (0-100)
config.timeout_seconds = 30;         // Execution timeout

// Security settings
config.allow_network = true;         // Allow network access
config.read_only_filesystem = false; // Read-only filesystem

// Platform-specific
config.base_image_path = "/path/to/base/image";  // Linux only
config.sandbox_id = "custom_id";     // Optional custom ID
```

### DRL Configuration

```cpp
// Model path
std::string model_path = "models/onnx/mtl_model.onnx";

// Database path
std::string db_path = "drlhss.db";

// Feature dimension (must match model)
int feature_dim = 16;

drl::DRLOrchestrator orchestrator(model_path, db_path, feature_dim);
```

## Performance Considerations

### Optimization Tips

1. **Sandbox Reuse**: Keep sandboxes initialized and reuse them
2. **Batch Processing**: Process multiple packets before sandbox analysis
3. **Async Processing**: Use background threads for sandbox execution
4. **Database Batching**: Batch database writes for better performance
5. **Model Caching**: Keep ONNX model loaded in memory

### Resource Requirements

| Component | CPU | Memory | Disk |
|-----------|-----|--------|------|
| DRL Inference | 10-20% | 100 MB | - |
| Linux Sandbox | 50% | 512 MB | 1 GB |
| Windows Sandbox | 50% | 512 MB | 1 GB |
| macOS Sandbox | 50% | 512 MB | 1 GB |
| Database | 5% | 50 MB | Variable |

## Troubleshooting

### Common Issues

#### 1. Sandbox Initialization Fails

**Linux:**
- Check if running with sufficient privileges (root or CAP_SYS_ADMIN)
- Verify cgroups v2 is available: `mount | grep cgroup2`
- Check seccomp support: `grep SECCOMP /boot/config-$(uname -r)`

**Windows:**
- Run as Administrator
- Verify AppContainer support (Windows 8+)
- Check Job Object creation permissions

**macOS:**
- Verify sandbox-exec is available
- Check TCC permissions
- Ensure code signing is not enforced for testing

#### 2. ONNX Model Loading Fails

- Verify model file exists and is readable
- Check ONNX Runtime version compatibility
- Ensure model input dimensions match feature extraction

#### 3. Database Errors

- Check write permissions on database file
- Verify SQLite3 is installed
- Check disk space availability

#### 4. High Memory Usage

- Reduce sandbox memory limits
- Decrease replay buffer size
- Enable database vacuuming

## Security Considerations

### Sandbox Escape Prevention

1. **Namespace Isolation** (Linux): Multiple namespace types prevent escape
2. **Resource Limits**: Prevent resource exhaustion attacks
3. **Syscall Filtering**: Block dangerous system calls
4. **Network Isolation**: Separate network namespace or filtering
5. **Filesystem Isolation**: Overlay/virtual filesystem prevents host modification

### Data Protection

1. **Hash Verification**: All artifacts are hashed for integrity
2. **Database Encryption**: Consider encrypting sensitive data
3. **Secure Cleanup**: Sandboxes are thoroughly cleaned after use
4. **Audit Logging**: All detections are logged to database

## Monitoring and Metrics

### Key Metrics

```cpp
auto stats = coordinator.getStats();

// Detection metrics
stats.total_detections
stats.network_detections
stats.file_detections
stats.behavior_detections
stats.malicious_detected
stats.false_positives

// Performance metrics
stats.avg_processing_time_ms
stats.nidps_stats.sandbox_executions

// Database metrics
stats.db_stats.telemetry_count
stats.db_stats.experience_count
stats.db_stats.pattern_count
```

### Logging

All components log to stdout/stderr with prefixes:
- `[UnifiedDetectionCoordinator]`
- `[NIDPSDetectionBridge]`
- `[LinuxSandbox]` / `[WindowsSandbox]` / `[MacOSSandbox]`
- `[DRLOrchestrator]`
- `[DatabaseManager]`

## Next Steps

1. Review [Cross-Platform Sandbox Architecture](CROSS_PLATFORM_SANDBOX_ARCHITECTURE.md)
2. Read [DRL Integration Guide](../DRL_FRAMEWORK_COMPLETE_REPORT.md)
3. Check [Deployment Guide](DEPLOYMENT_GUIDE.md)
4. See [API Reference](API_REFERENCE.md)

