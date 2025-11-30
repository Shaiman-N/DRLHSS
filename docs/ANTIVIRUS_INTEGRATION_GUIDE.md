# DRLHSS Antivirus Integration Guide

## Overview

The DRLHSS Antivirus system provides comprehensive malware detection through the integration of:
- **Static Analysis**: PE feature extraction and ML-based classification (2381 features)
- **Dynamic Analysis**: Runtime behavior monitoring (500 API call features)
- **DRL Enhancement**: Deep Reinforcement Learning for intelligent threat decisions
- **Sandbox Analysis**: Cross-platform isolated execution environment
- **Database Persistence**: SQLite-based threat intelligence storage

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   AVDetectionBridge                         │
│  (Orchestrates all AV components with DRLHSS integration)  │
└──────────────┬──────────────────────────────────────────────┘
               │
       ┌───────┴────────┬──────────────┬──────────────┐
       │                │              │              │
┌──────▼──────┐  ┌──────▼──────┐  ┌───▼────┐  ┌─────▼─────┐
│  AVService  │  │ ScanEngine  │  │  DRL   │  │  Sandbox  │
│  (Monitor)  │  │  (Analyze)  │  │Orchestr│  │  Factory  │
└──────┬──────┘  └──────┬──────┘  └───┬────┘  └─────┬─────┘
       │                │              │              │
┌──────▼──────────────────▼──────────────▼──────────────▼─────┐
│                    Database Manager                          │
│              (SQLite - Threat Intelligence)                  │
└──────────────────────────────────────────────────────────────┘
```

## Components

### 1. MalwareObject
Core detection object that represents a file/process under analysis.

**Lifecycle**:
1. Created → Static Analysis → Dynamic Analysis (optional) → DRL Inference → Completed → Terminated

**Features**:
- Unique object ID tracking
- Multi-phase analysis pipeline
- Threat level classification
- Comprehensive reporting

### 2. FeatureExtractor
Extracts 2381 PE features compatible with EMBER dataset.

**Feature Categories**:
- Byte histogram (256)
- Byte entropy histogram (256)
- String features (104)
- General info (10)
- Header features (62)
- Section features (255)
- Import features (1280)
- Export features (128)
- Data directory features (30)

### 3. BehaviorMonitor
Monitors runtime behavior and extracts 500 API call pattern features.

**Monitored Activities**:
- Network connections
- File operations
- Registry modifications
- Process creation
- Memory allocations
- Resource usage (CPU, memory)

### 4. InferenceEngine
Performs ML inference using ONNX Runtime.

**Models**:
- Static model: `antivirus_static_model.onnx` (2381 features)
- Dynamic model: `antivirus_dynamic_model.onnx` (500 features)

**Prediction**:
- Binary classification (benign/malicious)
- Confidence scores
- Hybrid prediction (60% static + 40% dynamic)

### 5. ScanEngine
Main scanning orchestrator that manages the analysis pipeline.

**Capabilities**:
- File scanning
- Process scanning
- Quarantine management
- Sandbox integration
- Statistics tracking

### 6. AVService
Background service for real-time protection.

**Features**:
- File system monitoring
- Scheduled scanning
- Multi-threaded processing
- Automatic quarantine
- Sandbox submission

### 7. AVDetectionBridge
Integration layer connecting AV with DRLHSS components.

**Integrations**:
- DRL Orchestrator: Intelligent threat decisions
- Sandbox Factory: Cross-platform execution
- Database Manager: Threat intelligence storage
- Telemetry conversion: File → DRL features

## Installation

### Prerequisites

```bash
# ONNX Runtime
# Windows: Download from https://github.com/microsoft/onnxruntime/releases
# Linux: sudo apt-get install libonnxruntime-dev
# macOS: brew install onnxruntime

# OpenSSL (for file hashing)
# Windows: vcpkg install openssl
# Linux: sudo apt-get install libssl-dev
# macOS: brew install openssl
```

### Build

```bash
cd DRLHSS
mkdir build && cd build

# Configure
cmake ..

# Build
cmake --build . --config Release

# Build specific target
cmake --build . --target av_integrated_example --config Release
```

## Usage

### Basic File Scanning

```cpp
#include "Detection/AVDetectionBridge.hpp"

// Configure
drlhss::detection::AVDetectionBridge::AVBridgeConfig config;
config.ml_model_path = "models/onnx/antivirus_static_model.onnx";
config.drl_model_path = "models/onnx/dqn_model.onnx";
config.enable_sandbox_analysis = true;
config.enable_drl_inference = true;

// Initialize
drlhss::detection::AVDetectionBridge av_bridge(config);
av_bridge.initialize();

// Scan file
auto result = av_bridge.scanFile("suspicious.exe");

if (result.is_malicious) {
    std::cout << "Threat detected: " << result.threat_classification << std::endl;
    std::cout << "Action: " << result.recommended_action << std::endl;
}
```

### Real-time Monitoring

```cpp
// Enable real-time monitoring
config.enable_real_time_monitoring = true;
config.scan_directories = {"/home", "/tmp", "/var"};

drlhss::detection::AVDetectionBridge av_bridge(config);
av_bridge.initialize();

// Set callback for detections
av_bridge.setScanCallback([](const auto& result) {
    if (result.is_malicious) {
        std::cout << "Real-time threat detected: " << result.file_path << std::endl;
    }
});

// Start monitoring
av_bridge.startMonitoring();

// ... monitoring runs in background ...

// Stop monitoring
av_bridge.stopMonitoring();
```

### Directory Scanning

```cpp
drlhss::detection::AVDetectionBridge av_bridge(config);
av_bridge.initialize();

// Scan entire directory
auto results = av_bridge.scanDirectory("/path/to/scan");

int threats = 0;
for (const auto& result : results) {
    if (result.is_malicious) {
        threats++;
        std::cout << "Threat: " << result.file_path << std::endl;
    }
}

std::cout << "Total threats found: " << threats << std::endl;
```

### With DRL Integration

```cpp
// DRL provides intelligent threat decisions
config.enable_drl_inference = true;

drlhss::detection::AVDetectionBridge av_bridge(config);
av_bridge.initialize();

auto result = av_bridge.scanFile("file.exe");

// DRL action: 0=Allow, 1=Block, 2=Quarantine, 3=DeepScan
std::cout << "DRL Action: " << result.drl_action << std::endl;
std::cout << "DRL Confidence: " << result.drl_confidence << "%" << std::endl;

// DRL Q-values for all actions
for (size_t i = 0; i < result.drl_q_values.size(); i++) {
    std::cout << "Q[" << i << "]: " << result.drl_q_values[i] << std::endl;
}
```

### With Sandbox Analysis

```cpp
// Sandbox provides behavioral analysis
config.enable_sandbox_analysis = true;

drlhss::detection::AVDetectionBridge av_bridge(config);
av_bridge.initialize();

auto result = av_bridge.scanFile("suspicious.exe");

if (result.sandbox_result.success) {
    std::cout << "Sandbox Analysis:" << std::endl;
    std::cout << "  Threat Score: " << result.sandbox_result.threat_score << "/100" << std::endl;
    std::cout << "  File Modified: " << result.sandbox_result.file_system_modified << std::endl;
    std::cout << "  Network Activity: " << result.sandbox_result.network_activity_detected << std::endl;
    std::cout << "  Process Created: " << result.sandbox_result.process_created << std::endl;
}
```

## Running Examples

### AV Integrated Example

```bash
# Scan current directory
./av_integrated_example

# Scan specific directory
./av_integrated_example /path/to/scan

# Real-time monitoring mode
./av_integrated_example /path/to/monitor --realtime
```

## Model Files

### Required ONNX Models

Place these files in `DRLHSS/models/onnx/`:

1. **antivirus_static_model.onnx**
   - Input: [1, 2381] float32
   - Output: [1, 2] float32 (benign, malicious scores)
   - Purpose: Static PE analysis

2. **antivirus_dynamic_model.onnx**
   - Input: [1, 500] float32
   - Output: [1, 2] float32 (benign, malicious scores)
   - Purpose: Dynamic behavior analysis

3. **dqn_model.onnx** (from DRL system)
   - Input: [1, 16] float32
   - Output: [1, 4] float32 (Q-values for 4 actions)
   - Purpose: DRL-based threat decisions

### Converting Models

If you have LightGBM models (.txt format):

```python
# Use the conversion script from Antivirus--final
python convert_models_to_onnx.py
```

## Configuration

### AVBridgeConfig Options

```cpp
struct AVBridgeConfig {
    std::string ml_model_path;              // Path to static ML model
    std::string drl_model_path;             // Path to DRL model
    std::string database_path;              // SQLite database path
    std::string quarantine_path;            // Quarantine directory
    std::vector<std::string> scan_directories;  // Monitored directories
    float malware_threshold;                // Detection threshold (0.0-1.0)
    bool enable_real_time_monitoring;       // Enable file system monitoring
    bool enable_sandbox_analysis;           // Enable sandbox execution
    bool enable_drl_inference;              // Enable DRL decisions
    bool enable_dynamic_analysis;           // Enable behavior monitoring
};
```

### Recommended Settings

**High Security**:
```cpp
config.malware_threshold = 0.6f;
config.enable_sandbox_analysis = true;
config.enable_drl_inference = true;
config.enable_dynamic_analysis = true;
```

**Balanced**:
```cpp
config.malware_threshold = 0.7f;
config.enable_sandbox_analysis = true;
config.enable_drl_inference = true;
config.enable_dynamic_analysis = false;
```

**Performance**:
```cpp
config.malware_threshold = 0.8f;
config.enable_sandbox_analysis = false;
config.enable_drl_inference = true;
config.enable_dynamic_analysis = false;
```

## Statistics and Monitoring

```cpp
auto stats = av_bridge.getStatistics();

std::cout << "Files Scanned: " << stats.files_scanned << std::endl;
std::cout << "Threats Detected: " << stats.threats_detected << std::endl;
std::cout << "Files Quarantined: " << stats.files_quarantined << std::endl;
std::cout << "Sandbox Analyses: " << stats.sandbox_analyses << std::endl;
std::cout << "DRL Inferences: " << stats.drl_inferences << std::endl;
std::cout << "Avg Scan Time: " << stats.avg_scan_time_ms << " ms" << std::endl;
```

## Threat Classification

The system classifies threats into the following categories:

- **CLEAN**: No malicious indicators
- **ML_DETECTED_MALWARE**: High ML confidence (>90%)
- **BEHAVIORAL_MALWARE**: Sandbox detected malicious behavior
- **SUSPICIOUS_EXECUTABLE**: Suspicious executable/script file
- **SUSPICIOUS_FILE**: General suspicious file

## Recommended Actions

Based on threat analysis, the system recommends:

- **ALLOW**: File is safe
- **MONITOR**: Low confidence threat, continue monitoring
- **QUARANTINE**: High confidence threat, isolate file
- **DELETE**: Critical threat (requires user confirmation)

## Performance Considerations

### Optimization Tips

1. **Disable Dynamic Analysis**: Saves ~15 seconds per file
2. **Adjust Sandbox Timeout**: Reduce from 60s to 30s for faster scans
3. **Batch Processing**: Scan multiple files concurrently
4. **Cache Results**: Store file hashes to avoid re-scanning

### Resource Usage

- **Memory**: ~500MB base + ~100MB per concurrent scan
- **CPU**: 1-4 cores depending on configuration
- **Disk**: ~10MB for models + quarantine space

## Troubleshooting

### Model Loading Fails

```
Error: Failed to load static model
```

**Solution**: Ensure ONNX models are in `models/onnx/` directory and ONNX Runtime is installed.

### Sandbox Not Available

```
Warning: Sandbox not available on this platform
```

**Solution**: Check platform support. Windows, Linux, and macOS are supported with different isolation mechanisms.

### High False Positive Rate

**Solution**: Adjust `malware_threshold` higher (e.g., 0.8 or 0.9) or enable DRL inference for better accuracy.

## Integration with Other DRLHSS Components

### With NIDPS

```cpp
// Create unified detection coordinator
drlhss::detection::UnifiedDetectionCoordinator coordinator(unified_config);
coordinator.initialize();
coordinator.startDetection();

// Handles both network and file threats
// Provides correlation analysis
```

### With DRL Training

```cpp
// AV detections feed into DRL training
// DRL learns from AV threat patterns
// Improves detection accuracy over time
```

### With Database

```cpp
// All detections stored in SQLite
// Query threat history
// Generate reports
```

## Security Considerations

1. **Quarantine Isolation**: Quarantined files are moved, not copied
2. **Sandbox Escape**: Sandboxes use OS-level isolation
3. **Model Integrity**: Verify ONNX model checksums
4. **Database Encryption**: Consider encrypting threat database
5. **Privilege Escalation**: Run with minimum required privileges

## Future Enhancements

- [ ] Cloud-based threat intelligence
- [ ] Automatic model updates
- [ ] Advanced PE parsing (full EMBER compatibility)
- [ ] Memory scanning for running processes
- [ ] Kernel-mode driver for deeper inspection
- [ ] Machine learning model retraining pipeline

## References

- EMBER Dataset: https://github.com/elastic/ember
- ONNX Runtime: https://onnxruntime.ai/
- LightGBM: https://lightgbm.readthedocs.io/
- PE Format: https://docs.microsoft.com/en-us/windows/win32/debug/pe-format

## Support

For issues or questions:
1. Check logs in `logs/av/`
2. Review database entries in `data/drlhss.db`
3. Examine quarantined files in `quarantine/`
4. Enable verbose logging for debugging

## License

Part of the DRLHSS project. See main LICENSE file.
