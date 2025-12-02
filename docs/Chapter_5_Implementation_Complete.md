# CHAPTER 5: IMPLEMENTATION

This chapter details the actual implementation of the DRLHSS system, covering development environment, coding practices, key implementation details, testing methodologies, and integration processes for all subsystems.

## 5.1 Development Environment and Tools

### 5.1.1 Programming Languages and Frameworks

**Primary Implementation**: C++17
- Modern C++ features (smart pointers, RAII, move semantics)
- Cross-platform compatibility
- High performance for real-time processing

**Training and Utilities**: Python 3.8+
- PyTorch for model training
- ONNX for model export
- NumPy, Pandas for data processing

**Build System**: CMake 3.15+
- Cross-platform build configuration
- Dependency management
- Multiple target support

### 5.1.2 Development Tools

**IDEs**: Visual Studio 2019+ (Windows), CLion, VS Code
**Version Control**: Git
**Compilers**: MSVC (Windows), GCC 9+ (Linux), Clang (macOS)
**Debuggers**: GDB, LLDB, Visual Studio Debugger

### 5.1.3 Key Libraries and Dependencies

**ML Inference**: ONNX Runtime 1.12+
**Network Capture**: libpcap 1.9+
**Database**: SQLite3 3.35+
**Cryptography**: OpenSSL 1.1+
**JSON Processing**: nlohmann/json 3.10+
**Testing**: Google Test (optional)

## 5.2 Overall System Implementation

### 5.2.1 Project Structure

```
DRLHSS/
├── include/          # Header files
│   ├── Detection/    # Detection layer headers
│   ├── DRL/          # DRL system headers
│   ├── Sandbox/      # Sandbox headers
│   ├── DB/           # Database headers
│   └── XAI/          # XAI headers
├── src/              # Implementation files
│   ├── Detection/    # Detection implementations
│   ├── DRL/          # DRL implementations
│   ├── Sandbox/      # Sandbox implementations
│   ├── DB/           # Database implementations
│   └── XAI/          # XAI implementations
├── models/onnx/      # ML models
├── python/           # Python utilities
├── tests/            # Test suites
└── CMakeLists.txt    # Build configuration
```

### 5.2.2 Build Configuration

**CMakeLists.txt** (Key sections):
```cmake
cmake_minimum_required(VERSION 3.15)
project(DRLHSS VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(onnxruntime REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(OpenSSL REQUIRED)

# Detection Layer
add_library(detection_layer
    src/Detection/NIDPS/*.cpp
    src/Detection/AV/*.cpp
    src/Detection/MD/*.cpp
)

# DRL System
add_library(drl_system
    src/DRL/DRLOrchestrator.cpp
    src/DRL/DRLInference.cpp
    src/DRL/EnvironmentAdapter.cpp
    src/DRL/ReplayBuffer.cpp
)

# Main executable
add_executable(drlhss_system
    src/main.cpp
)

target_link_libraries(drlhss_system
    detection_layer
    drl_system
    onnxruntime
    SQLite3::SQLite3
    OpenSSL::SSL
)
```

### 5.2.3 Compilation Process

**Windows**:
```cmd
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019"
cmake --build . --config Release
```

**Linux/macOS**:
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## 5.3 NIDPS Implementation

### 5.3.1 Packet Capture Implementation

**Key Code** (`PacketCapture.cpp`):
```cpp
class PacketCapture {
private:
    pcap_t* handle_;
    std::thread capture_thread_;
    std::atomic<bool> running_;
    
public:
    bool initialize(const std::string& interface) {
        char errbuf[PCAP_ERRBUF_SIZE];
        handle_ = pcap_open_live(
            interface.c_str(),
            BUFSIZ,
            1,  // promiscuous mode
            1000,  // timeout ms
            errbuf
        );
        
        if (!handle_) {
            std::cerr << "Error: " << errbuf << std::endl;
            return false;
        }
        
        return true;
    }
    
    void startCapture(PacketCallback callback) {
        running_ = true;
        capture_thread_ = std::thread([this, callback]() {
            pcap_loop(handle_, -1, callback, nullptr);
        });
    }
};
```

### 5.3.2 Flow Aggregation Implementation

**Flow Tracking**:
```cpp
struct FlowKey {
    uint32_t src_ip, dst_ip;
    uint16_t src_port, dst_port;
    uint8_t protocol;
    
    bool operator==(const FlowKey& other) const {
        return src_ip == other.src_ip &&
               dst_ip == other.dst_ip &&
               src_port == other.src_port &&
               dst_port == other.dst_port &&
               protocol == other.protocol;
    }
};

class FlowAggregator {
private:
    std::unordered_map<FlowKey, FlowStats> flows_;
    std::mutex mutex_;
    
public:
    void updateFlow(const Packet& packet) {
        std::lock_guard<std::mutex> lock(mutex_);
        FlowKey key = extractFlowKey(packet);
        flows_[key].update(packet);
    }
    
    std::vector<float> extractFeatures(const FlowKey& key) {
        auto& stats = flows_[key];
        return {
            stats.duration,
            stats.total_fwd_packets,
            stats.total_bwd_packets,
            // ... 41 features total
        };
    }
};
```

### 5.3.3 ML Model Inference

**ONNX Runtime Integration**:
```cpp
class NIDPSInference {
private:
    Ort::Env env_;
    Ort::Session session_;
    
public:
    NIDPSInference(const std::string& model_path) 
        : env_(ORT_LOGGING_LEVEL_WARNING, "NIDPS"),
          session_(env_, model_path.c_str(), Ort::SessionOptions()) {}
    
    std::pair<int, float> predict(const std::vector<float>& features) {
        // Prepare input tensor
        std::vector<int64_t> input_shape = {1, 41};
        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float*>(features.data()),
            features.size(), input_shape.data(), input_shape.size());
        
        // Run inference
        auto output_tensors = session_.Run(
            Ort::RunOptions{nullptr},
            input_names_.data(), &input_tensor, 1,
            output_names_.data(), 2);
        
        // Extract results
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        int prediction = std::distance(output_data, 
            std::max_element(output_data, output_data + 2));
        float confidence = output_data[prediction];
        
        return {prediction, confidence};
    }
};
```



## 5.4 Antivirus System Implementation

### 5.4.1 PE Feature Extraction

**Key Implementation** (`FeatureExtractor.cpp`):
```cpp
class FeatureExtractor {
public:
    std::vector<float> extractPEFeatures(const std::string& file_path) {
        std::vector<float> features(2381, 0.0f);
        
        // Read PE file
        std::ifstream file(file_path, std::ios::binary);
        std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)),
                                   std::istreambuf_iterator<char>());
        
        // Extract byte histogram (256 features)
        extractByteHistogram(data, features, 0);
        
        // Extract entropy features (256 features)
        extractEntropyFeatures(data, features, 256);
        
        // Extract string features (104 features)
        extractStringFeatures(data, features, 512);
        
        // Extract PE header features (62 features)
        extractPEHeaderFeatures(data, features, 616);
        
        // Extract section features (255 features)
        extractSectionFeatures(data, features, 678);
        
        // Extract import/export features (1448 features)
        extractImportExportFeatures(data, features, 933);
        
        return features;
    }
    
private:
    void extractByteHistogram(const std::vector<uint8_t>& data,
                              std::vector<float>& features, int offset) {
        std::array<int, 256> histogram = {0};
        for (uint8_t byte : data) {
            histogram[byte]++;
        }
        
        // Normalize
        for (int i = 0; i < 256; i++) {
            features[offset + i] = static_cast<float>(histogram[i]) / data.size();
        }
    }
};
```

### 5.4.2 Behavior Monitoring

**API Call Monitoring**:
```cpp
class BehaviorMonitor {
private:
    std::unordered_map<std::string, int> api_calls_;
    std::mutex mutex_;
    std::thread monitor_thread_;
    
public:
    void startMonitoring(DWORD process_id) {
        monitor_thread_ = std::thread([this, process_id]() {
            while (running_) {
                // Hook API calls (platform-specific)
                #ifdef _WIN32
                    monitorWindowsAPIs(process_id);
                #elif __linux__
                    monitorLinuxSyscalls(process_id);
                #endif
                
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
    }
    
    std::vector<float> extractBehaviorFeatures() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<float> features(500, 0.0f);
        
        // Map API calls to feature vector
        for (const auto& [api, count] : api_calls_) {
            int index = getAPIIndex(api);
            if (index >= 0 && index < 500) {
                features[index] = static_cast<float>(count);
            }
        }
        
        return features;
    }
};
```

### 5.4.3 Hybrid Prediction

**Combining Static and Dynamic**:
```cpp
class HybridPredictor {
public:
    DetectionResult predict(const std::string& file_path) {
        // Static analysis
        auto static_features = feature_extractor_.extractPEFeatures(file_path);
        auto [static_pred, static_conf] = static_model_.predict(static_features);
        
        // Dynamic analysis (optional)
        float dynamic_conf = 0.0f;
        if (enable_dynamic_) {
            auto dynamic_features = behavior_monitor_.extractBehaviorFeatures();
            auto [dynamic_pred, dyn_conf] = dynamic_model_.predict(dynamic_features);
            dynamic_conf = dyn_conf;
        }
        
        // Hybrid score: 60% static + 40% dynamic
        float hybrid_score = 0.6f * static_conf + 0.4f * dynamic_conf;
        
        return {
            .is_malicious = hybrid_score > threshold_,
            .confidence = hybrid_score,
            .static_score = static_conf,
            .dynamic_score = dynamic_conf
        };
    }
};
```

## 5.5 Malware Detection System Implementation

### 5.5.1 Multi-Stage Pipeline

**Pipeline Orchestration**:
```cpp
class MalwareProcessingPipeline {
public:
    DetectionResult processSample(const std::string& file_path) {
        // Stage 1: Initial Detection
        auto initial_result = malware_detector_.detect(file_path);
        
        if (initial_result.confidence < 0.5f) {
            return {.is_malicious = false, .stage = "initial"};
        }
        
        // Stage 2: Positive Sandbox (FP reduction)
        if (initial_result.is_malicious) {
            auto sandbox_result = positive_sandbox_.execute(file_path);
            if (!sandbox_result.malicious_behavior) {
                return {.is_malicious = false, .stage = "positive_sandbox"};
            }
        }
        
        // Stage 3: Negative Sandbox (FN detection)
        if (!initial_result.is_malicious) {
            auto sandbox_result = negative_sandbox_.execute(file_path);
            if (sandbox_result.malicious_behavior) {
                return {.is_malicious = true, .stage = "negative_sandbox"};
            }
        }
        
        // Stage 4: DRL Decision
        auto telemetry = convertToTelemetry(initial_result, sandbox_result);
        int action = drl_orchestrator_.processAndDecide(telemetry);
        
        return {
            .is_malicious = (action == 1 || action == 2),
            .action = action,
            .stage = "drl_decision"
        };
    }
};
```

### 5.5.2 Real-Time Monitoring

**File System Monitoring**:
```cpp
class RealTimeMonitor {
private:
    #ifdef _WIN32
    HANDLE directory_handle_;
    #elif __linux__
    int inotify_fd_;
    #endif
    
public:
    void startMonitoring(const std::string& path) {
        #ifdef _WIN32
        directory_handle_ = CreateFile(
            path.c_str(),
            FILE_LIST_DIRECTORY,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            NULL,
            OPEN_EXISTING,
            FILE_FLAG_BACKUP_SEMANTICS,
            NULL
        );
        
        monitor_thread_ = std::thread([this]() {
            BYTE buffer[4096];
            DWORD bytes_returned;
            
            while (running_) {
                if (ReadDirectoryChangesW(
                    directory_handle_,
                    buffer,
                    sizeof(buffer),
                    TRUE,
                    FILE_NOTIFY_CHANGE_FILE_NAME | FILE_NOTIFY_CHANGE_LAST_WRITE,
                    &bytes_returned,
                    NULL,
                    NULL)) {
                    processFileChanges(buffer, bytes_returned);
                }
            }
        });
        #elif __linux__
        inotify_fd_ = inotify_init();
        int wd = inotify_add_watch(inotify_fd_, path.c_str(),
                                    IN_CREATE | IN_MODIFY | IN_DELETE);
        
        monitor_thread_ = std::thread([this]() {
            char buffer[4096];
            while (running_) {
                int length = read(inotify_fd_, buffer, sizeof(buffer));
                if (length > 0) {
                    processInotifyEvents(buffer, length);
                }
            }
        });
        #endif
    }
};
```

## 5.6 DRL Framework Implementation

### 5.6.1 DRL Orchestrator

**Core Implementation**:
```cpp
class DRLOrchestrator {
private:
    std::unique_ptr<DRLInference> inference_;
    std::unique_ptr<EnvironmentAdapter> env_adapter_;
    std::unique_ptr<ReplayBuffer> replay_buffer_;
    std::unique_ptr<db::DatabaseManager> db_manager_;
    
public:
    bool initialize() {
        // Initialize ONNX inference
        inference_ = std::make_unique<DRLInference>(model_path_);
        
        // Initialize environment adapter
        env_adapter_ = std::make_unique<EnvironmentAdapter>(state_dim_);
        
        // Initialize replay buffer
        replay_buffer_ = std::make_unique<ReplayBuffer>(buffer_size_);
        
        // Initialize database
        db_manager_ = std::make_unique<db::DatabaseManager>(db_path_);
        db_manager_->initialize();
        
        return true;
    }
    
    int processAndDecide(const TelemetryData& telemetry) {
        // Convert telemetry to state vector
        auto state = env_adapter_->telemetryToState(telemetry);
        
        // Get Q-values from DRL model
        auto q_values = inference_->predict(state);
        
        // Select action (greedy)
        int action = std::distance(q_values.begin(),
            std::max_element(q_values.begin(), q_values.end()));
        
        // Store telemetry in database
        db_manager_->storeTelemetry(telemetry);
        
        return action;
    }
    
    void storeExperience(const TelemetryData& state,
                        int action,
                        float reward,
                        const TelemetryData& next_state,
                        bool done) {
        Experience exp{
            .state = env_adapter_->telemetryToState(state),
            .action = action,
            .reward = reward,
            .next_state = env_adapter_->telemetryToState(next_state),
            .done = done
        };
        
        replay_buffer_->add(exp);
        db_manager_->storeExperience(exp);
    }
};
```

### 5.6.2 Model Training Pipeline

**Python Training Script** (`train_complete.py`):
```python
import torch
import torch.nn as nn
import torch.optim as optim
import onnx
import numpy as np

class DQN(nn.Module):
    def __init__(self, state_dim=16, action_dim=4):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def train_dqn(num_episodes=10000):
    model = DQN()
    target_model = DQN()
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    
    for episode in range(num_episodes):
        # Training loop
        state = env.reset()
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(4)
            else:
                with torch.no_grad():
                    q_values = model(torch.FloatTensor(state))
                    action = q_values.argmax().item()
            
            next_state, reward, done = env.step(action)
            replay_buffer.add((state, action, reward, next_state, done))
            
            # Train on batch
            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                loss = compute_loss(model, target_model, batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            state = next_state
        
        # Update target network
        if episode % 100 == 0:
            target_model.load_state_dict(model.state_dict())
    
    # Export to ONNX
    dummy_input = torch.randn(1, 16)
    torch.onnx.export(model, dummy_input, "dqn_model.onnx",
                     input_names=['state'],
                     output_names=['q_values'])
```



## 5.7 Database System Implementation

### 5.7.1 Thread-Safe Database Manager

**Implementation** (`DatabaseManager.cpp`):
```cpp
class DatabaseManager {
private:
    sqlite3* db_;
    std::mutex mutex_;
    
public:
    bool initialize(const std::string& db_path) {
        int rc = sqlite3_open(db_path.c_str(), &db_);
        if (rc != SQLITE_OK) {
            return false;
        }
        
        // Create tables
        createTables();
        
        // Create indices for performance
        createIndices();
        
        return true;
    }
    
    bool storeTelemetry(const TelemetryData& telemetry) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        const char* sql = R"(
            INSERT INTO telemetry (
                sandbox_id, timestamp, syscall_count,
                file_operations, network_connections,
                cpu_usage, memory_usage, artifact_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        )";
        
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        
        sqlite3_bind_text(stmt, 1, telemetry.sandbox_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(stmt, 2, telemetry.timestamp);
        sqlite3_bind_int(stmt, 3, telemetry.syscall_count);
        sqlite3_bind_int(stmt, 4, telemetry.file_operations);
        sqlite3_bind_int(stmt, 5, telemetry.network_connections);
        sqlite3_bind_double(stmt, 6, telemetry.cpu_usage);
        sqlite3_bind_double(stmt, 7, telemetry.memory_usage);
        sqlite3_bind_text(stmt, 8, telemetry.artifact_hash.c_str(), -1, SQLITE_TRANSIENT);
        
        int rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        
        return rc == SQLITE_DONE;
    }
    
private:
    void createIndices() {
        executeSQL("CREATE INDEX IF NOT EXISTS idx_timestamp ON telemetry(timestamp)");
        executeSQL("CREATE INDEX IF NOT EXISTS idx_hash ON telemetry(artifact_hash)");
        executeSQL("CREATE INDEX IF NOT EXISTS idx_composite ON telemetry(timestamp, artifact_hash)");
    }
};
```

## 5.8 XAI System Implementation

### 5.8.1 Explanation Generator

**Implementation**:
```cpp
class ExplanationGenerator {
public:
    std::string generateExplanation(const DetectionEvent& event) {
        std::stringstream explanation;
        
        // Header
        explanation << "Threat Detection Report\n";
        explanation << "======================\n\n";
        
        // Basic information
        explanation << "File: " << event.file_path << "\n";
        explanation << "Detection Time: " << formatTimestamp(event.timestamp) << "\n";
        explanation << "Threat Level: " << getThreatLevel(event.confidence) << "\n\n";
        
        // Detection reasoning
        explanation << "Detection Reasoning:\n";
        explanation << generateReasoningText(event);
        
        // Feature importance
        explanation << "\nKey Indicators:\n";
        for (const auto& [feature, importance] : event.feature_importance) {
            if (importance > 0.1) {
                explanation << "  - " << feature << ": " 
                          << (importance * 100) << "% contribution\n";
            }
        }
        
        // Recommended action
        explanation << "\nRecommended Action: " << getActionText(event.action) << "\n";
        explanation << "Confidence: " << (event.confidence * 100) << "%\n";
        
        return explanation.str();
    }
    
private:
    std::string generateReasoningText(const DetectionEvent& event) {
        std::stringstream reasoning;
        
        if (event.static_score > 0.7) {
            reasoning << "  - Static analysis detected malicious PE characteristics\n";
        }
        
        if (event.dynamic_score > 0.7) {
            reasoning << "  - Behavioral analysis identified suspicious API patterns\n";
        }
        
        if (event.network_anomaly) {
            reasoning << "  - Unusual network communication patterns detected\n";
        }
        
        if (event.drl_confidence > 0.8) {
            reasoning << "  - AI model identified attack pattern similar to known threats\n";
        }
        
        return reasoning.str();
    }
};
```

### 5.8.2 Natural Language Interface

**Python NLP Engine** (`nlp_engine.py`):
```python
class NLPEngine:
    def __init__(self):
        self.templates = {
            'high_confidence': "High confidence malware detected. {details}",
            'medium_confidence': "Suspicious activity detected. {details}",
            'low_confidence': "Potential threat identified. {details}"
        }
    
    def generate_explanation(self, detection_data):
        confidence = detection_data['confidence']
        
        if confidence > 0.8:
            template = self.templates['high_confidence']
        elif confidence > 0.5:
            template = self.templates['medium_confidence']
        else:
            template = self.templates['low_confidence']
        
        details = self._generate_details(detection_data)
        return template.format(details=details)
    
    def _generate_details(self, data):
        details = []
        
        if data.get('static_malicious'):
            details.append("File structure matches known malware patterns")
        
        if data.get('dynamic_malicious'):
            details.append("Suspicious runtime behavior observed")
        
        if data.get('network_anomaly'):
            details.append("Abnormal network communication detected")
        
        return ". ".join(details)
```

## 5.9 Cross-Platform Sandbox Implementation

### 5.9.1 Linux Sandbox

**Implementation** (`LinuxSandbox.cpp`):
```cpp
class LinuxSandbox : public ISandbox {
public:
    SandboxResult execute(const std::string& file_path, int timeout_sec) override {
        pid_t pid = fork();
        
        if (pid == 0) {
            // Child process - set up sandbox
            setupNamespaces();
            setupCgroups();
            setupSeccomp();
            
            // Execute file
            execl(file_path.c_str(), file_path.c_str(), nullptr);
            exit(1);
        } else {
            // Parent process - monitor
            return monitorExecution(pid, timeout_sec);
        }
    }
    
private:
    void setupNamespaces() {
        // Create new namespaces for isolation
        unshare(CLONE_NEWPID | CLONE_NEWNET | CLONE_NEWNS | 
                CLONE_NEWIPC | CLONE_NEWUTS);
    }
    
    void setupCgroups() {
        // Set resource limits via cgroups
        std::ofstream cpu_limit("/sys/fs/cgroup/cpu/sandbox/cpu.cfs_quota_us");
        cpu_limit << "50000";  // 50% CPU
        
        std::ofstream mem_limit("/sys/fs/cgroup/memory/sandbox/memory.limit_in_bytes");
        mem_limit << "536870912";  // 512MB
    }
    
    void setupSeccomp() {
        // Install seccomp filter
        scmp_filter_ctx ctx = seccomp_init(SCMP_ACT_ALLOW);
        
        // Block dangerous syscalls
        seccomp_rule_add(ctx, SCMP_ACT_KILL, SCMP_SYS(ptrace), 0);
        seccomp_rule_add(ctx, SCMP_ACT_KILL, SCMP_SYS(reboot), 0);
        seccomp_rule_add(ctx, SCMP_ACT_KILL, SCMP_SYS(mount), 0);
        
        seccomp_load(ctx);
    }
};
```

### 5.9.2 Windows Sandbox

**Implementation** (`WindowsSandbox.cpp`):
```cpp
class WindowsSandbox : public ISandbox {
public:
    SandboxResult execute(const std::string& file_path, int timeout_sec) override {
        // Create job object for resource limits
        HANDLE job = CreateJobObject(NULL, NULL);
        
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits = {0};
        limits.BasicLimitInformation.LimitFlags = 
            JOB_OBJECT_LIMIT_PROCESS_MEMORY |
            JOB_OBJECT_LIMIT_JOB_MEMORY |
            JOB_OBJECT_LIMIT_ACTIVE_PROCESS;
        limits.ProcessMemoryLimit = 512 * 1024 * 1024;  // 512MB
        limits.JobMemoryLimit = 512 * 1024 * 1024;
        limits.BasicLimitInformation.ActiveProcessLimit = 1;
        
        SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                               &limits, sizeof(limits));
        
        // Create process with low integrity level
        STARTUPINFO si = {sizeof(si)};
        PROCESS_INFORMATION pi;
        
        if (CreateProcess(file_path.c_str(), NULL, NULL, NULL, FALSE,
                         CREATE_SUSPENDED, NULL, NULL, &si, &pi)) {
            // Assign to job
            AssignProcessToJobObject(job, pi.hProcess);
            
            // Set low integrity level
            setLowIntegrityLevel(pi.hProcess);
            
            // Resume and monitor
            ResumeThread(pi.hThread);
            return monitorExecution(pi.hProcess, timeout_sec);
        }
        
        return {.success = false};
    }
};
```

### 5.9.3 macOS Sandbox

**Implementation** (`MacOSSandbox.cpp`):
```cpp
class MacOSSandbox : public ISandbox {
public:
    SandboxResult execute(const std::string& file_path, int timeout_sec) override {
        // Create sandbox profile
        const char* profile = R"(
            (version 1)
            (deny default)
            (allow process-exec (literal "/path/to/file"))
            (allow file-read* (subpath "/usr/lib"))
            (allow file-read* (subpath "/System/Library"))
            (deny network*)
            (deny file-write*)
        )";
        
        pid_t pid = fork();
        
        if (pid == 0) {
            // Apply sandbox profile
            char* error;
            if (sandbox_init(kSBXProfilePureComputation, SANDBOX_NAMED, &error) != 0) {
                fprintf(stderr, "Sandbox init failed: %s\n", error);
                sandbox_free_error(error);
                exit(1);
            }
            
            // Execute file
            execl(file_path.c_str(), file_path.c_str(), nullptr);
            exit(1);
        } else {
            return monitorExecution(pid, timeout_sec);
        }
    }
};
```

## 5.10 Testing and Validation

### 5.10.1 Unit Testing

**Example Test** (`test_drl_inference.cpp`):
```cpp
#include <gtest/gtest.h>
#include "DRL/DRLInference.hpp"

class DRLInferenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        inference_ = std::make_unique<DRLInference>("test_model.onnx");
    }
    
    std::unique_ptr<DRLInference> inference_;
};

TEST_F(DRLInferenceTest, PredictReturnsValidAction) {
    std::vector<float> state(16, 0.5f);
    auto q_values = inference_->predict(state);
    
    ASSERT_EQ(q_values.size(), 4);
    
    int action = std::distance(q_values.begin(),
        std::max_element(q_values.begin(), q_values.end()));
    
    EXPECT_GE(action, 0);
    EXPECT_LT(action, 4);
}

TEST_F(DRLInferenceTest, HandlesInvalidInput) {
    std::vector<float> invalid_state(10, 0.0f);  // Wrong size
    
    EXPECT_THROW(inference_->predict(invalid_state), std::runtime_error);
}
```

### 5.10.2 Integration Testing

**System Integration Test**:
```cpp
TEST(IntegrationTest, EndToEndDetection) {
    // Initialize all components
    NIDPSDetectionBridge nidps(config);
    AVDetectionBridge av(config);
    MDDetectionBridge md(config);
    DRLOrchestrator drl(config);
    
    ASSERT_TRUE(nidps.initialize());
    ASSERT_TRUE(av.initialize());
    ASSERT_TRUE(md.initialize());
    ASSERT_TRUE(drl.initialize());
    
    // Test file detection
    auto result = av.scanFile("test_malware.exe");
    EXPECT_TRUE(result.is_malicious);
    EXPECT_GT(result.confidence, 0.7f);
    
    // Test DRL decision
    auto telemetry = convertToTelemetry(result);
    int action = drl.processAndDecide(telemetry);
    EXPECT_EQ(action, 1);  // Block action
}
```

### 5.10.3 Performance Testing

**Benchmark Tests**:
```cpp
void BenchmarkNIDPSInference() {
    NIDPSInference inference("mtl_model.onnx");
    std::vector<float> features(41, 0.5f);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 1000; i++) {
        inference.predict(features);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Average inference time: " 
              << (duration.count() / 1000.0) << " μs" << std::endl;
}
```

### 5.10.4 Debugging Practices

**Logging Implementation**:
```cpp
class Logger {
public:
    enum Level { DEBUG, INFO, WARNING, ERROR };
    
    static void log(Level level, const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        
        std::ofstream log_file("drlhss.log", std::ios::app);
        log_file << "[" << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S") << "] "
                 << levelToString(level) << ": " << message << std::endl;
    }
    
private:
    static std::mutex mutex_;
};
```

## 5.11 Deployment and Integration

### 5.11.1 Installation Process

**Installation Script** (`install.sh`):
```bash
#!/bin/bash

# Install dependencies
sudo apt-get update
sudo apt-get install -y libpcap-dev libsqlite3-dev libssl-dev

# Build system
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Install binaries
sudo make install

# Create data directories
sudo mkdir -p /var/lib/drlhss/models
sudo mkdir -p /var/lib/drlhss/data

# Copy models
sudo cp ../models/onnx/*.onnx /var/lib/drlhss/models/

# Set permissions
sudo chmod 755 /usr/local/bin/drlhss_system

echo "Installation complete!"
```

### 5.11.2 Configuration Management

**Configuration File** (`config.json`):
```json
{
  "nidps": {
    "interface": "eth0",
    "model_path": "/var/lib/drlhss/models/mtl_model.onnx",
    "threshold": 0.7
  },
  "antivirus": {
    "static_model": "/var/lib/drlhss/models/av_static.onnx",
    "dynamic_model": "/var/lib/drlhss/models/av_dynamic.onnx",
    "scan_threshold": 0.7,
    "enable_realtime": true
  },
  "malware_detection": {
    "dcnn_model": "/var/lib/drlhss/models/malware_dcnn.onnx",
    "malimg_model": "/var/lib/drlhss/models/malimg.onnx",
    "enable_sandbox": true
  },
  "drl": {
    "model_path": "/var/lib/drlhss/models/dqn_model.onnx",
    "state_dim": 16,
    "action_dim": 4
  },
  "database": {
    "path": "/var/lib/drlhss/data/drlhss.db"
  }
}
```

---

**Chapter 5 Summary**: This chapter detailed the complete implementation of DRLHSS, covering development environment setup, coding practices for all subsystems (NIDPS, AV, MD, DRL, DB, XAI, Sandboxes), testing methodologies, and deployment procedures. Key implementation details included packet capture, feature extraction, ML inference, multi-stage pipelines, cross-platform sandboxing, and comprehensive testing strategies. The system was implemented using modern C++17 with Python for training, achieving production-grade quality with proper error handling, thread safety, and performance optimization.

