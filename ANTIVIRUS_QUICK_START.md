# DRLHSS Antivirus - Quick Start Guide

## 5-Minute Setup

### 1. Prerequisites (2 minutes)

```bash
# Install ONNX Runtime
# Download from: https://github.com/microsoft/onnxruntime/releases
# Extract to: DRLHSS/external/onnxruntime/

# Install OpenSSL
# Windows: vcpkg install openssl
# Linux: sudo apt-get install libssl-dev
# macOS: brew install openssl
```

### 2. Build (2 minutes)

```bash
cd DRLHSS
mkdir build && cd build
cmake ..
cmake --build . --target av_integrated_example --config Release
```

### 3. Run (1 minute)

```bash
# Scan current directory
./av_integrated_example

# Scan specific directory
./av_integrated_example /path/to/scan

# Real-time monitoring
./av_integrated_example /path/to/monitor --realtime
```

## Basic Usage

### Scan a Single File

```cpp
#include "Detection/AVDetectionBridge.hpp"

// Configure
drlhss::detection::AVDetectionBridge::AVBridgeConfig config;
config.ml_model_path = "models/onnx/antivirus_static_model.onnx";
config.enable_drl_inference = true;
config.enable_sandbox_analysis = true;

// Initialize
drlhss::detection::AVDetectionBridge av_bridge(config);
av_bridge.initialize();

// Scan
auto result = av_bridge.scanFile("suspicious.exe");

// Check result
if (result.is_malicious) {
    std::cout << "⚠️ THREAT: " << result.threat_classification << std::endl;
    std::cout << "Action: " << result.recommended_action << std::endl;
} else {
    std::cout << "✅ CLEAN" << std::endl;
}
```

### Scan a Directory

```cpp
auto results = av_bridge.scanDirectory("/path/to/scan");

for (const auto& result : results) {
    if (result.is_malicious) {
        std::cout << "Threat: " << result.file_path << std::endl;
    }
}
```

### Real-time Monitoring

```cpp
config.enable_real_time_monitoring = true;
config.scan_directories = {"/home", "/tmp"};

drlhss::detection::AVDetectionBridge av_bridge(config);
av_bridge.initialize();

// Set callback
av_bridge.setScanCallback([](const auto& result) {
    if (result.is_malicious) {
        std::cout << "Real-time threat: " << result.file_path << std::endl;
    }
});

// Start monitoring
av_bridge.startMonitoring();

// ... runs in background ...

// Stop monitoring
av_bridge.stopMonitoring();
```

## Configuration Presets

### Maximum Security

```cpp
config.malware_threshold = 0.6f;
config.enable_sandbox_analysis = true;
config.enable_drl_inference = true;
config.enable_dynamic_analysis = true;
```

### Balanced (Recommended)

```cpp
config.malware_threshold = 0.7f;
config.enable_sandbox_analysis = true;
config.enable_drl_inference = true;
config.enable_dynamic_analysis = false;
```

### Fast Performance

```cpp
config.malware_threshold = 0.8f;
config.enable_sandbox_analysis = false;
config.enable_drl_inference = true;
config.enable_dynamic_analysis = false;
```

## Model Files

Place these in `DRLHSS/models/onnx/`:

1. `antivirus_static_model.onnx` - Static PE analysis
2. `antivirus_dynamic_model.onnx` - Dynamic behavior (optional)
3. `dqn_model.onnx` - DRL decisions (already present)

## Statistics

```cpp
auto stats = av_bridge.getStatistics();

std::cout << "Files Scanned: " << stats.files_scanned << std::endl;
std::cout << "Threats Detected: " << stats.threats_detected << std::endl;
std::cout << "Avg Scan Time: " << stats.avg_scan_time_ms << " ms" << std::endl;
```

## Command Line Examples

```bash
# Scan current directory
./av_integrated_example

# Scan Downloads folder
./av_integrated_example ~/Downloads

# Monitor /tmp in real-time
./av_integrated_example /tmp --realtime

# Full system with NIDPS + AV
./integrated_system_example
```

## Threat Actions

| Action | Meaning |
|--------|---------|
| ALLOW | File is safe |
| MONITOR | Low confidence, continue watching |
| QUARANTINE | High confidence threat, isolate |
| DELETE | Critical threat (requires confirmation) |

## DRL Actions

| Action | Value | Meaning |
|--------|-------|---------|
| Allow | 0 | File is benign |
| Block | 1 | Block file access |
| Quarantine | 2 | Move to quarantine |
| DeepScan | 3 | Send to sandbox |

## Troubleshooting

### Model Not Found
```
Error: Failed to load static model
Solution: Place .onnx files in models/onnx/ directory
```

### ONNX Runtime Error
```
Error: ONNX Runtime not found
Solution: Install ONNX Runtime and set ONNXRUNTIME_ROOT
```

### Sandbox Not Available
```
Warning: Sandbox not available
Solution: Check platform support (Windows/Linux/macOS)
```

## Performance Tips

1. **Disable Dynamic Analysis**: Saves ~15 seconds per file
2. **Adjust Threshold**: Higher = fewer false positives
3. **Batch Scanning**: Scan multiple files concurrently
4. **Cache Results**: Store hashes to avoid re-scanning

## Integration with NIDPS

```cpp
#include "Detection/UnifiedDetectionCoordinator.hpp"

// Unified protection (Network + Files)
drlhss::detection::UnifiedDetectionCoordinator::UnifiedConfig config;
config.enable_nidps = true;
config.enable_antivirus = true;

drlhss::detection::UnifiedDetectionCoordinator coordinator(config);
coordinator.initialize();
coordinator.startDetection();
```

## Next Steps

1. ✅ Build the system
2. ✅ Add ONNX models
3. ✅ Test with sample files
4. ✅ Configure thresholds
5. ✅ Deploy for production

## Full Documentation

See `docs/ANTIVIRUS_INTEGRATION_GUIDE.md` for complete documentation.

## Support

- Check logs in `logs/av/`
- Review database in `data/drlhss.db`
- Examine quarantine in `quarantine/`

---

**Ready to protect against real-world threats!** 🛡️
