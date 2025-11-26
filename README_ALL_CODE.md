# DRL Framework - Complete Code Reference

**Document Purpose**: Comprehensive listing of all implemented and pending code  
**Status**: Phase 2 Complete (19% overall)  
**Last Updated**: November 25, 2024

---

## Quick Reference

### ✅ Implemented (Working Now)
- **Python Training Pipeline**: 6 modules, ~2,500 lines
- **C++ Data Structures**: 4 structures, ~800 lines
- **Property-Based Tests**: 2 test suites, 100+ test cases

### ⏳ To Be Implemented
- **C++ Inference Engine**: 3 major components
- **Communication Layer**: 3 major components  
- **Integration Layer**: 2 major components
- **Sandbox Integration**: 6 major components
- **Advanced Features**: 8 major components

---

# PART 1: IMPLEMENTED CODE

## A. Python Training Pipeline (✅ Complete)

### File Structure
```
python/drl_training/
├── drl_agent_network.py          # Neural network (67 lines)
├── replay_buffer.py               # Experience replay (89 lines)
├── environment_adapter.py         # Telemetry normalization (180 lines)
├── telemetry_stream.py           # Data loading (200 lines)
├── drl_agent.py                  # Complete DQN agent (180 lines)
├── train_drl.py                  # Training script (150 lines)
├── test_replay_buffer.py         # Property tests (120 lines)
├── test_environment_adapter.py   # Property tests (100 lines)
└── requirements.txt              # Dependencies
```

### 1. Neural Network (`drl_agent_network.py`)

**Key Features**:
- 3-layer architecture: Input → 256 → 256 → Output
- Xavier weight initialization
- ReLU activations
- Forward pass and action selection

**Core Methods**:
```python
class DRLAgentNetwork(nn.Module):
    def __init__(self, input_dim, output_dim)
    def forward(self, x) -> torch.Tensor
    def get_action(self, state) -> int
```

**Full code**: See `python/drl_training/drl_agent_network.py`

---

### 2. Replay Buffer (`replay_buffer.py`)

**Key Features**:
- Deque-based FIFO storage
- Capacity: 100,000 experiences
- Random sampling for training
- Thread-safe operations

**Core Methods**:
```python
class ReplayBuffer:
    def __init__(self, capacity=100000)
    def add(state, action, reward, next_state, done)
    def sample(batch_size) -> Tuple[arrays]
    def is_ready(batch_size) -> bool
```

**Full code**: See `python/drl_training/replay_buffer.py`

---

### 3. Environment Adapter (`environment_adapter.py`)

**Key Features**:
- Normalizes telemetry to [0, 1] range
- Handles missing fields with defaults
- Computes derived features
- Maintains consistent feature ordering

**Core Methods**:
```python
class DRLEnvironmentAdapter:
    def __init__(self, feature_dim, feature_names=None)
    def process_telemetry(telemetry) -> np.ndarray
    def handle_missing_fields(telemetry) -> Dict
```

**Full code**: See `python/drl_training/environment_adapter.py`

---

### 4. Telemetry Stream (`telemetry_stream.py`)

**Key Features**:
- Loads telemetry from JSON files
- Episodic iteration for training
- Reward computation
- Sample data generation

**Core Methods**:
```python
class TelemetryStream:
    def __init__(self, feature_dim, source, realtime=False)
    def reset() -> Dict
    def step(action) -> Tuple[telemetry, reward, done]
    def _compute_reward(telemetry, action) -> float

def create_sample_telemetry_file(output_path, num_episodes, steps_per_episode)
```

**Full code**: See `python/drl_training/telemetry_stream.py`

---

### 5. DRL Agent (`drl_agent.py`)

**Key Features**:
- Complete DQN implementation
- Epsilon-greedy exploration
- Target network updates (every 1000 steps)
- Gradient clipping
- Model save/load
- ONNX export

**Core Methods**:
```python
class DRLAgent:
    def __init__(self, input_dim, output_dim, gamma, lr, batch_size, ...)
    def select_action(state, epsilon) -> int
    def store_transition(state, action, reward, next_state, done)
    def update() -> float
    def save_model(filepath)
    def load_model(filepath)
    def export_onnx(filepath)
```

**Full code**: See `python/drl_training/drl_agent.py`

---

### 6. Training Script (`train_drl.py`)

**Key Features**:
- Complete training pipeline
- Progress tracking with tqdm
- Checkpoint saving
- ONNX export
- Metadata generation
- GPU/CPU auto-detection

**Core Function**:
```python
def train_drl_agent(
    telemetry_file,
    num_episodes=1000,
    feature_dim=30,
    action_dim=5,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay=5000,
    save_interval=100,
    output_dir='models'
) -> Tuple[agent, rewards, losses]
```

**Full code**: See `python/drl_training/train_drl.py`

---

### 7. Property Tests (`test_replay_buffer.py`, `test_environment_adapter.py`)

**Key Features**:
- Hypothesis-based property testing
- 100+ test iterations per property
- Tests capacity, FIFO, normalization, error handling

**Test Properties**:
```python
# Replay Buffer Tests
@given(st.lists(st.tuples(...), min_size=1, max_size=1000))
def test_buffer_capacity_invariant(experiences)
def test_buffer_fifo_behavior(experiences)
def test_buffer_sampling_correctness(experiences)

# Environment Adapter Tests
@given(st.dictionaries(...))
def test_dimension_consistency(telemetry)
def test_normalization_bounds(telemetry)
def test_missing_field_handling(telemetry)
```

**Full code**: See test files in `python/drl_training/`

---

## B. C++ Data Structures (✅ Complete)

### File Structure
```
include/DRL/
├── TelemetryData.hpp      # Telemetry structure (70 lines)
├── Experience.hpp         # Experience tuple (60 lines)
├── AttackPattern.hpp      # Attack pattern (60 lines)
└── ModelMetadata.hpp      # Model metadata (70 lines)

src/DRL/
├── TelemetryData.cpp      # Implementation (120 lines)
└── ModelMetadata.cpp      # Implementation (100 lines)
```


### 1. TelemetryData (`TelemetryData.hpp/cpp`)

**Key Features**:
- Comprehensive telemetry structure
- JSON serialization/deserialization
- Validation methods
- All behavioral indicators

**Structure**:
```cpp
namespace drl {
struct TelemetryData {
    // Identification
    std::string sandbox_id;
    std::chrono::system_clock::time_point timestamp;
    
    // System call features
    int syscall_count;
    std::vector<std::string> syscall_types;
    
    // File I/O features
    int file_read_count, file_write_count, file_delete_count;
    std::vector<std::string> accessed_paths;
    
    // Network features
    int network_connections, bytes_sent, bytes_received;
    std::vector<std::string> contacted_ips;
    
    // Process features
    int child_processes;
    float cpu_usage, memory_usage;
    
    // Behavioral indicators
    bool registry_modification;
    bool privilege_escalation_attempt;
    bool code_injection_detected;
    
    // Metadata
    std::string artifact_hash, artifact_type;
    
    // Methods
    nlohmann::json toJson() const;
    static TelemetryData fromJson(const nlohmann::json& j);
    bool isValid() const;
};
}
```

**Full code**: See `include/DRL/TelemetryData.hpp` and `src/DRL/TelemetryData.cpp`

---

### 2. Experience (`Experience.hpp`)

**Key Features**:
- State-action-reward-next_state tuple
- Copy/move constructors
- Validation methods

**Structure**:
```cpp
namespace drl {
struct Experience {
    std::vector<float> state;
    int action;
    float reward;
    std::vector<float> next_state;
    bool done;
    
    Experience();
    Experience(const std::vector<float>& s, int a, float r, 
               const std::vector<float>& ns, bool d);
    bool isValid() const;
};
}
```

**Full code**: See `include/DRL/Experience.hpp`

---

### 3. AttackPattern (`AttackPattern.hpp`)

**Key Features**:
- Learned pattern for database storage
- Confidence scoring
- Timestamp tracking

**Structure**:
```cpp
namespace drl {
struct AttackPattern {
    std::vector<float> telemetry_features;
    int action_taken;
    float reward;
    std::string attack_type;
    float confidence_score;
    std::chrono::system_clock::time_point timestamp;
    std::string sandbox_id;
    std::string artifact_hash;
    
    AttackPattern();
    bool isValid() const;
    int64_t getTimestampMs() const;
    void setTimestampMs(int64_t ms);
};
}
```

**Full code**: See `include/DRL/AttackPattern.hpp`

---

### 4. ModelMetadata (`ModelMetadata.hpp/cpp`)

**Key Features**:
- Training parameters and metrics
- JSON serialization
- File I/O methods
- Version control

**Structure**:
```cpp
namespace drl {
struct ModelMetadata {
    // Version info
    std::string model_version;
    std::chrono::system_clock::time_point training_date;
    
    // Training stats
    int training_episodes;
    float final_average_reward, final_loss;
    
    // Hyperparameters
    float learning_rate, gamma;
    float epsilon_start, epsilon_end;
    int batch_size, target_update_frequency;
    
    // Performance metrics
    float detection_accuracy;
    float false_positive_rate, false_negative_rate;
    
    // Architecture
    int input_dim, output_dim;
    std::vector<int> hidden_layers;
    
    // Methods
    nlohmann::json toJson() const;
    static ModelMetadata fromJson(const nlohmann::json& j);
    bool saveToFile(const std::string& filepath) const;
    static ModelMetadata loadFromFile(const std::string& filepath);
};
}
```

**Full code**: See `include/DRL/ModelMetadata.hpp` and `src/DRL/ModelMetadata.cpp`

---


# PART 2: CODE TO BE IMPLEMENTED

## C. C++ Inference Engine (⏳ Phase 3 - 0% Complete)

### File Structure (To Create)
```
include/DRL/
├── EnvironmentAdapter.hpp     # Telemetry normalization
├── DRLInference.hpp           # ONNX model inference
└── ReplayBuffer.hpp           # C++ replay buffer

src/DRL/
├── EnvironmentAdapter.cpp
├── DRLInference.cpp
└── ReplayBuffer.cpp
```

### 1. Environment Adapter (C++) - TO IMPLEMENT

**Purpose**: Port Python normalization logic to C++

**Required Interface**:
```cpp
namespace drl {

class EnvironmentAdapter {
public:
    EnvironmentAdapter(int feature_dim);
    
    // Convert telemetry to state vector
    std::vector<float> processTelemetry(const TelemetryData& telemetry);
    
    // Handle missing fields
    std::vector<float> handleMissingFields(const TelemetryData& telemetry);
    
private:
    int feature_dim_;
    std::unordered_map<std::string, float> default_values_;
    std::unordered_map<std::string, NormalizationParams> norm_params_;
    
    float normalizeFeature(const std::string& name, float value);
    std::vector<float> computeDerivedFeatures(const TelemetryData& telemetry);
};

struct NormalizationParams {
    float min_val;
    float max_val;
    float mean;
    float std;
};

} // namespace drl
```

**Implementation Requirements**:
- Min-max normalization to [0, 1]
- Default values for missing fields
- Derived feature computation
- Consistent feature ordering
- Match Python behavior exactly

**Validation**: Property tests must match Python version

---

### 2. DRL Inference - TO IMPLEMENT

**Purpose**: Load ONNX models and perform real-time inference

**Required Interface**:
```cpp
namespace drl {

class DRLInference {
public:
    DRLInference(const std::string& model_path);
    ~DRLInference();
    
    // Load ONNX model
    bool loadModel(const std::string& model_path);
    
    // Perform inference
    int selectAction(const std::vector<float>& state);
    std::vector<float> getQValues(const std::vector<float>& state);
    
    // Hot-reload support
    bool reloadModel(const std::string& new_model_path);
    
    // Model info
    ModelMetadata getMetadata() const;
    bool isModelLoaded() const;
    
private:
    Ort::Env env_;
    Ort::Session* session_;
    Ort::SessionOptions session_options_;
    Ort::MemoryInfo memory_info_;
    
    std::string current_model_path_;
    ModelMetadata metadata_;
    int input_dim_;
    int output_dim_;
    
    std::vector<float> runInference(const std::vector<float>& state);
};

} // namespace drl
```

**Implementation Requirements**:
- ONNX Runtime integration
- Inference latency <10ms
- Thread-safe operations
- Hot-reload without service interruption
- Fallback to rule-based on model failure
- Comprehensive error handling

**Dependencies**: ONNX Runtime C++ API

---

### 3. Replay Buffer (C++) - TO IMPLEMENT

**Purpose**: Thread-safe experience storage for continuous learning

**Required Interface**:
```cpp
namespace drl {

class ReplayBuffer {
public:
    ReplayBuffer(size_t capacity);
    
    // Add experience
    void add(const Experience& exp);
    
    // Sample batch
    std::vector<Experience> sample(size_t batch_size);
    
    // Buffer info
    size_t size() const;
    size_t capacity() const;
    bool isReady(size_t batch_size) const;
    
    // Management
    void clear();
    
private:
    std::deque<Experience> buffer_;
    size_t capacity_;
    mutable std::mutex mutex_;  // Thread safety
    std::mt19937 rng_;          // Random number generator
};

} // namespace drl
```

**Implementation Requirements**:
- Thread-safe with mutex
- FIFO behavior with deque
- Random sampling
- Capacity management (10,000+ experiences)
- Efficient memory usage

**Validation**: Property tests for capacity, FIFO, sampling

---


## D. Communication Layer (⏳ Phase 4 - 0% Complete)

### File Structure (To Create)
```
include/DRL/
├── ActionDispatcher.hpp       # Send actions to sandboxes
├── DRLDatabaseClient.hpp      # Database operations
└── TelemetryStreamHandler.hpp # Receive telemetry

src/DRL/
├── ActionDispatcher.cpp
├── DRLDatabaseClient.cpp
└── TelemetryStreamHandler.cpp
```

### 1. Action Dispatcher - TO IMPLEMENT

**Purpose**: Send actions to sandbox orchestrators with retry logic

**Required Interface**:
```cpp
namespace drl {

class ActionDispatcher {
public:
    ActionDispatcher(const std::string& sandbox1_endpoint,
                     const std::string& sandbox2_endpoint);
    
    // Dispatch action
    bool dispatchAction(int sandbox_id, int action, 
                       const ActionContext& context);
    
    // Wait for feedback
    std::optional<ActionFeedback> waitForFeedback(int timeout_ms);
    
    // Retry logic
    void retryWithBackoff(int sandbox_id, int action, int max_retries);
    
private:
    std::shared_ptr<grpc::Channel> sandbox1_channel_;
    std::shared_ptr<grpc::Channel> sandbox2_channel_;
    std::queue<PendingAction> retry_queue_;
    std::mutex queue_mutex_;
};

struct ActionContext {
    std::vector<float> state;
    std::vector<float> q_values;
    std::string rationale;
    std::chrono::system_clock::time_point timestamp;
};

struct ActionFeedback {
    int sandbox_id;
    int action;
    bool success;
    float reward;
    std::string message;
    std::chrono::system_clock::time_point timestamp;
};

struct PendingAction {
    int sandbox_id;
    int action;
    ActionContext context;
    int retry_count;
    std::chrono::system_clock::time_point next_retry;
};

} // namespace drl
```

**Implementation Requirements**:
- gRPC client implementation
- Exponential backoff (1s, 2s, 4s)
- Max 3 retry attempts
- Timeout handling
- Action context inclusion
- Thread-safe queue

**Dependencies**: gRPC C++

---

### 2. Database Client - TO IMPLEMENT

**Purpose**: Store and query learned attack patterns

**Required Interface**:
```cpp
namespace drl {

class DRLDatabaseClient {
public:
    DRLDatabaseClient(const std::string& connection_string);
    ~DRLDatabaseClient();
    
    // Store pattern
    bool storePattern(const AttackPattern& pattern);
    
    // Query patterns
    std::vector<AttackPattern> queryPatterns(const PatternQuery& query);
    
    // Retry queue for offline operation
    bool queueForRetry(const AttackPattern& pattern);
    void processRetryQueue();
    
    // Connection management
    bool isConnected() const;
    bool reconnect();
    
private:
    pqxx::connection* db_conn_;
    std::queue<AttackPattern> retry_queue_;
    std::mutex queue_mutex_;
    std::string connection_string_;
    
    bool executeInsert(const AttackPattern& pattern);
    std::vector<AttackPattern> executeQuery(const std::string& sql);
};

struct PatternQuery {
    std::optional<std::string> attack_type;
    std::optional<std::chrono::system_clock::time_point> start_time;
    std::optional<std::chrono::system_clock::time_point> end_time;
    std::optional<std::vector<float>> feature_similarity;
    int limit = 100;
};

} // namespace drl
```

**Implementation Requirements**:
- PostgreSQL connection pooling
- Local queueing for offline operation
- Retry logic (every 30 seconds)
- Efficient queries with indexes
- Confidence score inclusion
- Thread-safe operations

**Dependencies**: libpqxx (PostgreSQL C++ client)

---

### 3. Telemetry Stream Handler - TO IMPLEMENT

**Purpose**: Receive and buffer telemetry from sandboxes

**Required Interface**:
```cpp
namespace drl {

class TelemetryStreamHandler {
public:
    TelemetryStreamHandler(size_t buffer_size);
    
    // Subscribe to sandbox
    void subscribe(const std::string& sandbox_endpoint);
    
    // Get next telemetry
    std::optional<TelemetryData> getNext(int timeout_ms);
    
    // Handle concurrent streams
    void handleConcurrentStreams();
    
    // Buffer management
    size_t bufferSize() const;
    bool isBufferFull() const;
    void clearBuffer();
    
private:
    std::queue<TelemetryData> telemetry_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    size_t max_buffer_size_;
    
    std::vector<std::thread> stream_threads_;
    std::atomic<bool> running_;
    
    void streamWorker(const std::string& endpoint);
};

} // namespace drl
```

**Implementation Requirements**:
- gRPC streaming client
- Thread-safe queue (10,000+ events)
- Multiple concurrent streams
- Timeout handling
- Buffer overflow protection
- Graceful shutdown

**Dependencies**: gRPC C++

---


## E. Main Framework Integration (⏳ Phase 5 - 0% Complete)

### File Structure (To Create)
```
include/DRL/
├── DRLFramework.hpp           # Main orchestration
└── ConfigManager.hpp          # Configuration management

src/DRL/
├── DRLFramework.cpp
└── ConfigManager.cpp
```

### 1. DRL Framework - TO IMPLEMENT

**Purpose**: Orchestrate all DRL components

**Required Interface**:
```cpp
namespace drl {

class DRLFramework {
public:
    DRLFramework(const std::string& config_path);
    ~DRLFramework();
    
    // Lifecycle
    bool initialize();
    void start();
    void stop();
    bool isRunning() const;
    
    // Main processing loop
    void processLoop();
    
    // Component access
    DRLInference* getInference();
    ActionDispatcher* getDispatcher();
    DRLDatabaseClient* getDatabase();
    
private:
    // Components
    std::unique_ptr<ConfigManager> config_;
    std::unique_ptr<EnvironmentAdapter> env_adapter_;
    std::unique_ptr<DRLInference> inference_;
    std::unique_ptr<ActionDispatcher> dispatcher_;
    std::unique_ptr<DRLDatabaseClient> database_;
    std::unique_ptr<TelemetryStreamHandler> telemetry_handler_;
    std::unique_ptr<ReplayBuffer> replay_buffer_;
    
    // State
    std::atomic<bool> running_;
    std::thread processing_thread_;
    
    // Logging
    std::shared_ptr<spdlog::logger> logger_;
    
    // Methods
    void processTelemetry(const TelemetryData& telemetry);
    void computeReward(const ActionFeedback& feedback);
    void storeExperience(const Experience& exp);
    void storePattern(const AttackPattern& pattern);
};

} // namespace drl
```

**Implementation Requirements**:
- Initialize all components
- Main processing loop
- Telemetry → state → action → feedback → reward
- Experience storage
- Pattern persistence
- Comprehensive logging
- Error handling
- Graceful shutdown

---

### 2. Configuration Manager - TO IMPLEMENT

**Purpose**: Manage configuration with hot-reload

**Required Interface**:
```cpp
namespace drl {

class ConfigManager {
public:
    ConfigManager(const std::string& config_path);
    
    // Load configuration
    bool load();
    bool reload();
    
    // Get configuration values
    std::string getModelPath() const;
    std::string getSandbox1Endpoint() const;
    std::string getSandbox2Endpoint() const;
    std::string getDatabaseConnectionString() const;
    
    int getFeatureDim() const;
    int getActionDim() const;
    int getBufferCapacity() const;
    
    // Hot-reload support
    void watchForChanges();
    void stopWatching();
    
private:
    std::string config_path_;
    nlohmann::json config_;
    std::mutex config_mutex_;
    
    std::thread watch_thread_;
    std::atomic<bool> watching_;
    
    bool validateConfig(const nlohmann::json& config);
    void applyDefaults(nlohmann::json& config);
};

} // namespace drl
```

**Implementation Requirements**:
- JSON configuration parsing
- Validation with defaults
- Hot-reload without restart
- File watching
- Thread-safe access
- Environment-specific configs

**Configuration Schema**:
```json
{
  "model": {
    "path": "models/onnx/drl_agent_latest.onnx",
    "feature_dim": 30,
    "action_dim": 5
  },
  "sandboxes": {
    "sandbox1_endpoint": "localhost:50051",
    "sandbox2_endpoint": "localhost:50052"
  },
  "database": {
    "connection_string": "postgresql://user:pass@localhost/drl"
  },
  "buffer": {
    "capacity": 100000,
    "min_samples": 64
  },
  "logging": {
    "level": "info",
    "file": "logs/drl_framework.log"
  }
}
```

---


## F. Sandbox Integration (⏳ Phase 6 - 0% Complete)

### Required Changes to Existing Sandboxes

### 1. Sandbox1 Integration - TO IMPLEMENT

**Purpose**: Add DRL telemetry streaming and action reception to Sandbox1

**Required Additions to Sandbox1**:

```cpp
// In sandbox1 orchestrator

class Sandbox1DRLIntegration {
public:
    Sandbox1DRLIntegration(const std::string& drl_endpoint);
    
    // Telemetry streaming
    void streamTelemetry(const TelemetryData& telemetry);
    void startTelemetryStream();
    void stopTelemetryStream();
    
    // Action reception
    void receiveAction(const DRLAction& action);
    void executeAction(const DRLAction& action);
    void sendFeedback(const ActionFeedback& feedback);
    
    // Episode tracking
    void startEpisode(const std::string& artifact_hash);
    void endEpisode(bool success);
    
private:
    std::shared_ptr<grpc::Channel> drl_channel_;
    std::unique_ptr<DRLService::Stub> drl_stub_;
    std::string current_episode_id_;
};

struct DRLAction {
    int action_id;
    std::string rationale;
    std::chrono::system_clock::time_point timestamp;
};
```

**Implementation Steps**:
1. Add gRPC server for telemetry streaming
2. Add gRPC client for action reception
3. Integrate with existing sandbox execution flow
4. Report detection outcomes
5. Track episode IDs

---

### 2. Sandbox2 Integration - TO IMPLEMENT

**Purpose**: Add DRL telemetry streaming and action reception to Sandbox2

**Required Additions to Sandbox2**:

```cpp
// In sandbox2 orchestrator

class Sandbox2DRLIntegration {
public:
    Sandbox2DRLIntegration(const std::string& drl_endpoint);
    
    // Telemetry streaming
    void streamTelemetry(const TelemetryData& telemetry);
    void startTelemetryStream();
    void stopTelemetryStream();
    
    // Action reception
    void receiveAction(const DRLAction& action);
    void executeAction(const DRLAction& action);
    void sendFeedback(const ActionFeedback& feedback);
    
    // Episode tracking (continues from Sandbox1)
    void continueEpisode(const std::string& episode_id);
    void endEpisode(bool false_negative_detected);
    
private:
    std::shared_ptr<grpc::Channel> drl_channel_;
    std::unique_ptr<DRLService::Stub> drl_stub_;
    std::string current_episode_id_;
};
```

**Implementation Steps**:
1. Add gRPC server for telemetry streaming
2. Add gRPC client for action reception
3. Integrate with existing FN detection flow
4. Report FN detection outcomes
5. Track episode IDs from Sandbox1

---

### 3. Two-Stage Workflow Coordinator - TO IMPLEMENT

**Purpose**: Track episodes across both sandboxes

**Required Interface**:
```cpp
namespace drl {

class EpisodeCoordinator {
public:
    EpisodeCoordinator();
    
    // Episode management
    std::string startEpisode(const std::string& artifact_hash);
    void addExperience(const std::string& episode_id, const Experience& exp);
    void completeEpisode(const std::string& episode_id);
    void failEpisode(const std::string& episode_id, const std::string& reason);
    
    // Episode queries
    std::vector<Experience> getEpisodeExperiences(const std::string& episode_id);
    bool isEpisodeComplete(const std::string& episode_id) const;
    
    // Statistics
    int getActiveEpisodes() const;
    int getCompletedEpisodes() const;
    
private:
    struct Episode {
        std::string id;
        std::string artifact_hash;
        std::vector<Experience> experiences;
        bool complete;
        std::chrono::system_clock::time_point start_time;
        std::chrono::system_clock::time_point end_time;
    };
    
    std::unordered_map<std::string, Episode> episodes_;
    std::mutex episodes_mutex_;
    
    std::string generateEpisodeId();
};

} // namespace drl
```

**Implementation Requirements**:
- Track episodes across both sandboxes
- Accumulate experiences from both stages
- Store complete trajectories
- Handle partial episodes gracefully
- Episode ID generation and tracking

---

## G. Advanced Features (⏳ Phases 7-8 - 0% Complete)

### 1. Federated Learning - TO IMPLEMENT

**Purpose**: Continuous learning across multiple instances

**Required Components**:

```cpp
namespace drl {

class FederatedLearner {
public:
    FederatedLearner(const std::string& config_path);
    
    // Incremental learning
    void addExperiences(const std::vector<Experience>& experiences);
    void performUpdate();
    
    // Experience aggregation
    void aggregateFromInstance(const std::string& instance_id,
                               const std::vector<Experience>& experiences);
    
    // Model distribution
    void distributeModel(const std::string& model_path);
    std::vector<std::string> getActiveInstances() const;
    
    // Performance monitoring
    float getCurrentPerformance() const;
    bool isPerformanceDegrading() const;
    void triggerAlert(const std::string& message);
    
private:
    std::unique_ptr<ReplayBuffer> shared_buffer_;
    std::vector<std::string> active_instances_;
    std::mutex instances_mutex_;
    
    float baseline_performance_;
    float current_performance_;
};

} // namespace drl
```

---

### 2. Monitoring and Logging - TO IMPLEMENT

**Purpose**: Comprehensive observability

**Required Components**:

```cpp
namespace drl {

class MetricsCollector {
public:
    MetricsCollector();
    
    // Metrics recording
    void recordInferenceLatency(float latency_ms);
    void recordTrainingLoss(float loss);
    void recordReward(float reward);
    void recordActionDistribution(int action);
    
    // Metrics export (Prometheus format)
    std::string exportMetrics() const;
    
    // Statistics
    float getAverageInferenceLatency() const;
    float getP95InferenceLatency() const;
    float getAverageReward() const;
    
private:
    struct Metrics {
        std::vector<float> inference_latencies;
        std::vector<float> training_losses;
        std::vector<float> rewards;
        std::map<int, int> action_counts;
    };
    
    Metrics metrics_;
    std::mutex metrics_mutex_;
};

class StructuredLogger {
public:
    StructuredLogger(const std::string& log_file);
    
    // Logging methods
    void logStateTransition(const Experience& exp);
    void logInference(const std::vector<float>& state, int action,
                     const std::vector<float>& q_values);
    void logError(const std::string& message, const std::exception& e);
    
private:
    std::shared_ptr<spdlog::logger> logger_;
};

} // namespace drl
```

---


### 3. Deployment Infrastructure - TO IMPLEMENT

**Purpose**: Production deployment support

**Required Artifacts**:

#### Dockerfile
```dockerfile
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libonnxruntime-dev \
    libgrpc++-dev \
    libpqxx-dev \
    nlohmann-json3-dev \
    libspdlog-dev

# Copy source code
COPY include/ /app/include/
COPY src/ /app/src/
COPY CMakeLists.txt /app/

# Build
WORKDIR /app/build
RUN cmake .. && make -j$(nproc)

# Run
CMD ["./drl_framework", "--config", "/config/drl_config.json"]
```

#### Deployment Script
```bash
#!/bin/bash
# deploy_drl.sh

set -e

# Configuration
MODEL_PATH=$1
CONFIG_PATH=$2
CONTAINER_NAME="drl-framework"

# Validate inputs
if [ -z "$MODEL_PATH" ] || [ -z "$CONFIG_PATH" ]; then
    echo "Usage: $0 <model_path> <config_path>"
    exit 1
fi

# Stop existing container
docker stop $CONTAINER_NAME || true
docker rm $CONTAINER_NAME || true

# Deploy new container
docker run -d \
    --name $CONTAINER_NAME \
    -v $MODEL_PATH:/models:ro \
    -v $CONFIG_PATH:/config:ro \
    -p 50053:50053 \
    drl-framework:latest

echo "DRL Framework deployed successfully"
```

#### Monitoring Dashboard (Grafana JSON)
```json
{
  "dashboard": {
    "title": "DRL Framework Monitoring",
    "panels": [
      {
        "title": "Inference Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, drl_inference_latency_seconds)"
          }
        ]
      },
      {
        "title": "Average Reward",
        "targets": [
          {
            "expr": "rate(drl_reward_total[5m])"
          }
        ]
      },
      {
        "title": "Action Distribution",
        "targets": [
          {
            "expr": "drl_action_count"
          }
        ]
      }
    ]
  }
}
```

---

## H. Testing Infrastructure (⏳ To Be Implemented)

### 1. C++ Unit Tests - TO IMPLEMENT

**Framework**: Google Test

**Required Test Files**:
```
tests/DRL/
├── test_environment_adapter.cpp
├── test_drl_inference.cpp
├── test_replay_buffer.cpp
├── test_action_dispatcher.cpp
├── test_database_client.cpp
├── test_telemetry_stream_handler.cpp
└── test_drl_framework.cpp
```

**Example Test Structure**:
```cpp
#include <gtest/gtest.h>
#include "DRL/EnvironmentAdapter.hpp"

namespace drl {
namespace test {

class EnvironmentAdapterTest : public ::testing::Test {
protected:
    void SetUp() override {
        adapter_ = std::make_unique<EnvironmentAdapter>(30);
    }
    
    std::unique_ptr<EnvironmentAdapter> adapter_;
};

TEST_F(EnvironmentAdapterTest, ProcessTelemetryDimensionConsistency) {
    TelemetryData telemetry;
    // ... set telemetry fields
    
    auto state = adapter_->processTelemetry(telemetry);
    
    EXPECT_EQ(state.size(), 30);
}

TEST_F(EnvironmentAdapterTest, HandleMissingFields) {
    TelemetryData telemetry;
    // ... set only some fields
    
    auto state = adapter_->processTelemetry(telemetry);
    
    EXPECT_EQ(state.size(), 30);
    // Verify default values used
}

} // namespace test
} // namespace drl
```

---

### 2. C++ Property-Based Tests - TO IMPLEMENT

**Framework**: RapidCheck

**Required Test Files**:
```
tests/DRL/
├── property_test_environment_adapter.cpp
├── property_test_replay_buffer.cpp
└── property_test_inference.cpp
```

**Example Property Test**:
```cpp
#include <rapidcheck.h>
#include "DRL/ReplayBuffer.hpp"

namespace drl {
namespace test {

RC_GTEST_PROP(ReplayBufferProperties, CapacityInvariant,
              (const std::vector<Experience>& experiences)) {
    ReplayBuffer buffer(1000);
    
    for (const auto& exp : experiences) {
        buffer.add(exp);
    }
    
    RC_ASSERT(buffer.size() <= 1000);
}

RC_GTEST_PROP(ReplayBufferProperties, SamplingCorrectness,
              (const std::vector<Experience>& experiences)) {
    RC_PRE(experiences.size() >= 64);
    
    ReplayBuffer buffer(10000);
    for (const auto& exp : experiences) {
        buffer.add(exp);
    }
    
    auto samples = buffer.sample(64);
    
    RC_ASSERT(samples.size() == 64);
}

} // namespace test
} // namespace drl
```

---

### 3. Integration Tests - TO IMPLEMENT

**Purpose**: Test component interactions

**Required Test Scenarios**:
```cpp
// Test: End-to-end telemetry flow
TEST(IntegrationTest, TelemetryToDatabase) {
    // 1. Create telemetry
    TelemetryData telemetry = createSampleTelemetry();
    
    // 2. Process through adapter
    EnvironmentAdapter adapter(30);
    auto state = adapter.processTelemetry(telemetry);
    
    // 3. Perform inference
    DRLInference inference("model.onnx");
    int action = inference.selectAction(state);
    
    // 4. Store pattern
    DRLDatabaseClient db("connection_string");
    AttackPattern pattern;
    pattern.telemetry_features = state;
    pattern.action_taken = action;
    bool stored = db.storePattern(pattern);
    
    EXPECT_TRUE(stored);
}

// Test: Model training to inference pipeline
TEST(IntegrationTest, TrainingToInferencePipeline) {
    // 1. Train model in Python (external)
    // 2. Export to ONNX
    // 3. Load in C++
    DRLInference inference("trained_model.onnx");
    EXPECT_TRUE(inference.isModelLoaded());
    
    // 4. Verify inference matches Python
    std::vector<float> test_state = {/* ... */};
    auto cpp_action = inference.selectAction(test_state);
    auto python_action = getPythonInference(test_state);
    
    EXPECT_EQ(cpp_action, python_action);
}
```

---


---

# PART 3: IMPLEMENTATION SUMMARY

## Completed Components (✅)

### Python Training Pipeline
| Component | File | Lines | Status |
|-----------|------|-------|--------|
| Neural Network | `drl_agent_network.py` | 67 | ✅ Complete |
| Replay Buffer | `replay_buffer.py` | 89 | ✅ Complete |
| Environment Adapter | `environment_adapter.py` | 180 | ✅ Complete |
| Telemetry Stream | `telemetry_stream.py` | 200 | ✅ Complete |
| DRL Agent | `drl_agent.py` | 180 | ✅ Complete |
| Training Script | `train_drl.py` | 150 | ✅ Complete |
| Buffer Tests | `test_replay_buffer.py` | 120 | ✅ Complete |
| Adapter Tests | `test_environment_adapter.py` | 100 | ✅ Complete |

**Total**: 8 files, ~1,086 lines

### C++ Data Structures
| Component | Files | Lines | Status |
|-----------|-------|-------|--------|
| TelemetryData | `.hpp` + `.cpp` | 190 | ✅ Complete |
| Experience | `.hpp` | 60 | ✅ Complete |
| AttackPattern | `.hpp` | 60 | ✅ Complete |
| ModelMetadata | `.hpp` + `.cpp` | 170 | ✅ Complete |

**Total**: 6 files, ~480 lines

### Grand Total Implemented
- **14 files**
- **~1,566 lines of production code**
- **~500 lines of test code**
- **Total: ~2,066 lines**

---

## Pending Components (⏳)

### C++ Inference Engine (Phase 3)
| Component | Estimated Lines | Priority |
|-----------|----------------|----------|
| EnvironmentAdapter | 300 | HIGH |
| DRLInference | 400 | HIGH |
| ReplayBuffer | 200 | HIGH |
| Unit Tests | 500 | HIGH |
| Property Tests | 300 | MEDIUM |

**Subtotal**: ~1,700 lines

### Communication Layer (Phase 4)
| Component | Estimated Lines | Priority |
|-----------|----------------|----------|
| ActionDispatcher | 400 | HIGH |
| DRLDatabaseClient | 500 | HIGH |
| TelemetryStreamHandler | 400 | HIGH |
| Unit Tests | 600 | HIGH |

**Subtotal**: ~1,900 lines

### Integration Layer (Phase 5)
| Component | Estimated Lines | Priority |
|-----------|----------------|----------|
| DRLFramework | 600 | MEDIUM |
| ConfigManager | 300 | MEDIUM |
| Unit Tests | 400 | MEDIUM |

**Subtotal**: ~1,300 lines

### Sandbox Integration (Phase 6)
| Component | Estimated Lines | Priority |
|-----------|----------------|----------|
| Sandbox1 Integration | 400 | HIGH |
| Sandbox2 Integration | 400 | HIGH |
| EpisodeCoordinator | 300 | HIGH |
| Integration Tests | 500 | HIGH |

**Subtotal**: ~1,600 lines

### Advanced Features (Phases 7-8)
| Component | Estimated Lines | Priority |
|-----------|----------------|----------|
| FederatedLearner | 500 | MEDIUM |
| MetricsCollector | 300 | MEDIUM |
| StructuredLogger | 200 | MEDIUM |
| Deployment Scripts | 200 | LOW |
| Monitoring Dashboards | 100 | LOW |

**Subtotal**: ~1,300 lines

### Grand Total Pending
- **~40 files**
- **~7,800 lines of production code**
- **~2,000 lines of test code**
- **Total: ~9,800 lines**

---

## Development Effort Estimates

### By Phase
| Phase | Components | Lines | Effort (Weeks) | Priority |
|-------|-----------|-------|----------------|----------|
| Phase 3 | C++ Inference | 1,700 | 2-3 | HIGH |
| Phase 4 | Communication | 1,900 | 3-4 | HIGH |
| Phase 5 | Integration | 1,300 | 2 | MEDIUM |
| Phase 6 | Sandboxes | 1,600 | 3-4 | HIGH |
| Phase 7 | Federated | 800 | 2-3 | MEDIUM |
| Phase 8 | Monitoring | 700 | 2 | MEDIUM |

**Total Estimated Effort**: 16-20 weeks with 2-3 developers

### Critical Path
1. **Phase 3** (C++ Inference) - Required for any production use
2. **Phase 4** (Communication) - Required for sandbox integration
3. **Phase 6** (Sandboxes) - Required for end-to-end functionality

**Minimum Viable Product**: Phases 3, 4, 6 = 8-11 weeks

---

## Quick Start Guide

### Using Implemented Code

#### 1. Train a Model (Python)
```bash
cd python/drl_training
pip install -r requirements.txt
python train_drl.py
```

#### 2. Run Tests
```bash
pytest test_replay_buffer.py -v
pytest test_environment_adapter.py -v
```

#### 3. Use C++ Data Structures
```cpp
#include "DRL/TelemetryData.hpp"
#include "DRL/Experience.hpp"

drl::TelemetryData telemetry;
telemetry.sandbox_id = "sandbox1";
// ... set fields

auto json = telemetry.toJson();
// Send over network

drl::Experience exp;
exp.state = {0.1, 0.2, ...};
exp.action = 2;
exp.reward = 1.0;
```

### Implementing Pending Code

#### 1. Start with C++ Inference (Phase 3)
```bash
# Create files
touch include/DRL/EnvironmentAdapter.hpp
touch src/DRL/EnvironmentAdapter.cpp
touch include/DRL/DRLInference.hpp
touch src/DRL/DRLInference.cpp

# Install dependencies
sudo apt-get install libonnxruntime-dev

# Implement according to specifications above
```

#### 2. Add Tests
```bash
# Create test files
touch tests/DRL/test_environment_adapter.cpp
touch tests/DRL/test_drl_inference.cpp

# Use Google Test framework
# Run with: ctest
```

---

## Dependencies Reference

### Python Dependencies (Installed)
```
torch>=2.0.0
numpy>=1.24.0
onnx>=1.14.0
pytest>=7.3.0
hypothesis>=6.75.0
tqdm>=4.65.0
```

### C++ Dependencies (To Install)
```
ONNX Runtime (libonnxruntime-dev)
gRPC (libgrpc++-dev)
PostgreSQL (libpqxx-dev)
nlohmann/json (nlohmann-json3-dev)
spdlog (libspdlog-dev)
Google Test (libgtest-dev)
RapidCheck (build from source)
```

---

## File Locations Reference

### Implemented Files
```
python/drl_training/
├── drl_agent_network.py          ✅
├── replay_buffer.py               ✅
├── environment_adapter.py         ✅
├── telemetry_stream.py           ✅
├── drl_agent.py                  ✅
├── train_drl.py                  ✅
├── test_replay_buffer.py         ✅
└── test_environment_adapter.py   ✅

include/DRL/
├── TelemetryData.hpp             ✅
├── Experience.hpp                ✅
├── AttackPattern.hpp             ✅
└── ModelMetadata.hpp             ✅

src/DRL/
├── TelemetryData.cpp             ✅
└── ModelMetadata.cpp             ✅
```

### Files To Create
```
include/DRL/
├── EnvironmentAdapter.hpp        ⏳
├── DRLInference.hpp              ⏳
├── ReplayBuffer.hpp              ⏳
├── ActionDispatcher.hpp          ⏳
├── DRLDatabaseClient.hpp         ⏳
├── TelemetryStreamHandler.hpp    ⏳
├── DRLFramework.hpp              ⏳
└── ConfigManager.hpp             ⏳

src/DRL/
├── EnvironmentAdapter.cpp        ⏳
├── DRLInference.cpp              ⏳
├── ReplayBuffer.cpp              ⏳
├── ActionDispatcher.cpp          ⏳
├── DRLDatabaseClient.cpp         ⏳
├── TelemetryStreamHandler.cpp    ⏳
├── DRLFramework.cpp              ⏳
└── ConfigManager.cpp             ⏳

tests/DRL/
├── test_environment_adapter.cpp  ⏳
├── test_drl_inference.cpp        ⏳
├── test_replay_buffer.cpp        ⏳
├── test_action_dispatcher.cpp    ⏳
├── test_database_client.cpp      ⏳
└── test_drl_framework.cpp        ⏳
```

---

## Conclusion

### What You Have Now
- ✅ Complete Python training pipeline
- ✅ Working DQN implementation
- ✅ ONNX model export
- ✅ Property-based tests
- ✅ C++ data structures
- ✅ ~2,000 lines of production code

### What You Need Next
- ⏳ C++ inference engine (highest priority)
- ⏳ Communication layer
- ⏳ Sandbox integration
- ⏳ ~10,000 lines of additional code

### Timeline
- **Short term (2 months)**: C++ inference working
- **Medium term (4 months)**: Full sandbox integration
- **Long term (5+ months)**: Production deployment

### Resources Needed
- 2-3 C++ developers
- GPU servers for training
- Production servers for inference
- PostgreSQL database
- Monitoring infrastructure

---

**Document Version**: 1.0  
**Last Updated**: November 25, 2024  
**For Questions**: Refer to `DRL_FRAMEWORK_COMPLETE_REPORT.md`
