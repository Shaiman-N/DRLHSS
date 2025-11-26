# Design Document: DRL Framework for Cybersecurity Platform

## Overview

The DRL (Deep Reinforcement Learning) Framework is a production-grade system that learns attack patterns and adaptive containment strategies from real-time sandbox telemetry. The framework consists of two main components:

1. **Python Training Pipeline**: Runs on Google Colab or similar environments to train DRL models using real telemetry data, then exports trained models to ONNX format
2. **C++ Inference Engine**: Loads ONNX models and performs real-time inference on live telemetry streams, dispatching actions to sandbox orchestrators and persisting learned patterns to the database

The system implements a continuous learning loop where experiences from both sandbox1 (positive/false positive detection) and sandbox2 (negative/false negative detection) are used to improve detection policies over time, following a federated learning approach.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DRL Framework                                │
│                                                                   │
│  ┌──────────────────────┐         ┌──────────────────────┐     │
│  │  Python Training     │         │   C++ Inference      │     │
│  │     Pipeline         │────────▶│      Engine          │     │
│  │                      │  ONNX   │                      │     │
│  │  - Telemetry Replay  │  Model  │  - Model Loader      │     │
│  │  - DQN Training      │         │  - Real-time Infer   │     │
│  │  - Experience Replay │         │  - Action Dispatch   │     │
│  │  - ONNX Export       │         │  - DB Integration    │     │
│  └──────────────────────┘         └──────────────────────┘     │
│           ▲                                 │                    │
│           │                                 ▼                    │
│  ┌────────┴─────────┐         ┌────────────────────┐          │
│  │  Telemetry       │         │  Action Dispatcher │          │
│  │  Data Store      │         │                    │          │
│  └──────────────────┘         └────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
         ▲                                    │
         │                                    ▼
┌────────┴────────┐                  ┌───────────────┐
│   Sandbox1      │                  │   Sandbox2    │
│  (Positive FP)  │                  │ (Negative FN) │
└─────────────────┘                  └───────────────┘
         │                                    │
         └────────────────┬───────────────────┘
                          ▼
                  ┌───────────────┐
                  │   Database    │
                  │ (Learned      │
                  │  Patterns)    │
                  └───────────────┘
```

### Component Interaction Flow

1. **Telemetry Collection**: Sandbox orchestrators generate telemetry during file execution
2. **State Normalization**: Environment Adapter converts raw telemetry to fixed-dimension state vectors
3. **Action Selection**: Policy network performs inference and selects optimal action
4. **Action Execution**: Action Dispatcher sends commands to sandbox orchestrators
5. **Reward Computation**: Feedback from sandboxes is converted to reward signals
6. **Experience Storage**: State-action-reward-next_state tuples stored in replay buffer
7. **Model Training**: Periodic training updates using experience replay and target networks
8. **Pattern Persistence**: Learned attack patterns stored in database for detection layer
9. **Model Export**: Trained models exported to ONNX for C++ deployment
10. **Continuous Learning**: New telemetry continuously improves the policy

## Components and Interfaces

### 1. Environment Adapter

**Purpose**: Normalizes raw telemetry data into fixed-dimension state vectors suitable for neural network input.

**Interface**:
```cpp
class EnvironmentAdapter {
public:
    EnvironmentAdapter(int feature_dim);
    std::vector<float> processTelementry(const TelemetryData& raw_telemetry);
    std::vector<float> handleMissingFields(const TelemetryData& telemetry);
private:
    int feature_dim_;
    std::unordered_map<std::string, float> default_values_;
};
```

**Responsibilities**:
- Parse raw telemetry JSON/protobuf data
- Extract relevant features (syscalls, file I/O, network activity, etc.)
- Normalize values to [0, 1] range
- Handle missing or malformed fields with defaults
- Maintain consistent feature ordering

### 2. Policy Network (DQN)

**Purpose**: Neural network that estimates Q-values for state-action pairs to guide decision-making.

**Architecture**:
- Input Layer: Feature dimension (e.g., 30 telemetry features)
- Hidden Layer 1: 256 neurons, ReLU activation
- Hidden Layer 2: 256 neurons, ReLU activation
- Output Layer: Action dimension (e.g., 5 discrete actions)

**Actions**:
- 0: Continue monitoring (no intervention)
- 1: Increase isolation level
- 2: Terminate suspicious process
- 3: Quarantine file immediately
- 4: Request human review

### 3. Experience Replay Buffer

**Purpose**: Stores past experiences to break temporal correlations and stabilize training.

**Interface**:
```cpp
class ReplayBuffer {
public:
    ReplayBuffer(size_t capacity);
    void add(const Experience& exp);
    std::vector<Experience> sample(size_t batch_size);
    size_t size() const;
private:
    std::deque<Experience> buffer_;
    size_t capacity_;
};

struct Experience {
    std::vector<float> state;
    int action;
    float reward;
    std::vector<float> next_state;
    bool done;
};
```

### 4. Policy Learner

**Purpose**: Implements DQN training algorithm with target network stabilization.

**Interface**:
```python
class PolicyLearner:
    def __init__(self, input_dim, output_dim, learning_rate, gamma):
        self.policy_net = DRLAgentNetwork(input_dim, output_dim)
        self.target_net = DRLAgentNetwork(input_dim, output_dim)
        self.optimizer = Adam(learning_rate)
        self.gamma = gamma
        self.update_counter = 0
        
    def update(self, batch):
        # Compute Q-values and target Q-values
        # Perform gradient descent
        # Update target network every N steps
        pass
```

### 5. Action Dispatcher

**Purpose**: Sends learned actions to sandbox orchestrators and manages feedback loop.

**Interface**:
```cpp
class ActionDispatcher {
public:
    ActionDispatcher(const std::string& sandbox1_endpoint,
                     const std::string& sandbox2_endpoint);
    bool dispatchAction(int sandbox_id, int action, const ActionContext& context);
    std::optional<ActionFeedback> waitForFeedback(int timeout_ms);
    void retryWithBackoff(int sandbox_id, int action, int max_retries);
private:
    grpc::Channel sandbox1_channel_;
    grpc::Channel sandbox2_channel_;
    std::queue<PendingAction> retry_queue_;
};
```

### 6. Database Client

**Purpose**: Persists learned attack patterns and retrieves them for detection systems.

**Interface**:
```cpp
class DRLDatabaseClient {
public:
    DRLDatabaseClient(const std::string& connection_string);
    bool storePattern(const AttackPattern& pattern);
    std::vector<AttackPattern> queryPatterns(const PatternQuery& query);
    bool queueForRetry(const AttackPattern& pattern);
private:
    pqxx::connection db_conn_;
    std::queue<AttackPattern> retry_queue_;
};

struct AttackPattern {
    std::vector<float> telemetry_features;
    int action_taken;
    float reward;
    std::string attack_type;
    float confidence_score;
    std::chrono::system_clock::time_point timestamp;
};
```

### 7. Telemetry Stream Handler

**Purpose**: Manages real-time telemetry ingestion from sandbox orchestrators.

**Interface**:
```cpp
class TelemetryStreamHandler {
public:
    TelemetryStreamHandler(size_t buffer_size);
    void subscribe(const std::string& sandbox_endpoint);
    std::optional<TelemetryData> getNext(int timeout_ms);
    void handleConcurrentStreams();
private:
    std::queue<TelemetryData> telemetry_queue_;
    std::mutex queue_mutex_;
    size_t max_buffer_size_;
};
```

## Data Models

### Telemetry Data Structure

```cpp
struct TelemetryData {
    std::string sandbox_id;
    std::chrono::system_clock::time_point timestamp;
    
    // System call features
    int syscall_count;
    std::vector<std::string> syscall_types;
    
    // File I/O features
    int file_read_count;
    int file_write_count;
    int file_delete_count;
    std::vector<std::string> accessed_paths;
    
    // Network features
    int network_connections;
    std::vector<std::string> contacted_ips;
    int bytes_sent;
    int bytes_received;
    
    // Process features
    int child_processes;
    float cpu_usage;
    float memory_usage;
    
    // Behavioral indicators
    bool registry_modification;
    bool privilege_escalation_attempt;
    bool code_injection_detected;
    
    // Metadata
    std::string artifact_hash;
    std::string artifact_type;
};
```

### Model Metadata

```cpp
struct ModelMetadata {
    std::string model_version;
    std::chrono::system_clock::time_point training_date;
    int training_episodes;
    float final_average_reward;
    float final_loss;
    
    // Hyperparameters
    float learning_rate;
    float gamma;
    float epsilon_start;
    float epsilon_end;
    int batch_size;
    int target_update_frequency;
    
    // Performance metrics
    float detection_accuracy;
    float false_positive_rate;
    float false_negative_rate;
};
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Telemetry Reception Latency

*For any* telemetry event generated by a sandbox orchestrator, the DRL Framework should receive and process it within 100 milliseconds.

**Validates: Requirements 1.1**

### Property 2: State Vector Dimension Consistency

*For any* raw telemetry input, the Environment Adapter should produce a state vector of exactly the configured feature dimension.

**Validates: Requirements 1.2**

### Property 3: Graceful Error Handling for Malformed Telemetry

*For any* telemetry data with missing or malformed fields, the Environment Adapter should not crash and should fill missing values with configured defaults.

**Validates: Requirements 1.3**

### Property 4: Concurrent Stream Processing Without Data Loss

*For any* set of concurrent telemetry streams, all telemetry events should be processed without loss or corruption.

**Validates: Requirements 1.4**

### Property 5: Replay Buffer Capacity Invariant

*For any* sequence of operations, the replay buffer should maintain capacity for at least 10,000 observations.

**Validates: Requirements 1.5**

### Property 6: Experience Storage Completeness

*For any* valid state-action-reward-next_state tuple, it should be successfully stored in the replay buffer.

**Validates: Requirements 2.1**

### Property 7: Training Trigger Condition

*For any* replay buffer state with at least 64 samples, a training update should be triggered.

**Validates: Requirements 2.2**

### Property 8: Target Network Update Frequency

*For any* training session, the target network should be updated exactly every 1000 gradient descent steps.

**Validates: Requirements 2.3**

### Property 9: Positive Reward for Successful Detection

*For any* successful malware detection event, the assigned reward value should be positive.

**Validates: Requirements 2.4**

### Property 10: Negative Reward for Detection Errors

*For any* false positive or false negative event, the assigned reward value should be negative.

**Validates: Requirements 2.5**

### Property 11: ONNX Export on Training Completion

*For any* completed training session, an ONNX model file should be created in the models directory.

**Validates: Requirements 3.1**

### Property 12: ONNX Model Round-Trip Validity

*For any* exported ONNX model, loading it with ONNX Runtime should succeed without errors.

**Validates: Requirements 3.2**

### Property 13: Model Metadata Completeness

*For any* exported model, the metadata should include all required training parameters and performance metrics.

**Validates: Requirements 3.3**

### Property 14: Model Version Timestamp Ordering

*For any* sequence of model exports, the timestamps should be monotonically increasing.

**Validates: Requirements 3.4**

### Property 15: Model Preservation on Export Failure

*For any* failed model export operation, the previous valid model should remain unchanged.

**Validates: Requirements 3.5**

### Property 16: Inference Latency Bound

*For any* state vector input, inference should complete within 10 milliseconds.

**Validates: Requirements 4.2**

### Property 17: Argmax Action Selection

*For any* Q-value output from inference, the selected action should be the one with maximum Q-value.

**Validates: Requirements 4.3**

### Property 18: Hot-Reload Service Continuity

*For any* model update during operation, inference requests should continue to be served without interruption.

**Validates: Requirements 4.5**

### Property 19: Action Dispatch Completeness

*For any* action selected by the DRL agent, a dispatch command should be sent to the appropriate sandbox orchestrator.

**Validates: Requirements 5.1**

### Property 20: Action Context Inclusion

*For any* dispatched action, the message should include context about the decision rationale.

**Validates: Requirements 5.2**

### Property 21: Feedback Loop Closure

*For any* action executed by a sandbox, confirmation feedback should be received by the DRL framework.

**Validates: Requirements 5.3**

### Property 22: Reward Computation from Feedback

*For any* feedback received from a sandbox, a reward signal should be computed for learning.

**Validates: Requirements 5.4**

### Property 23: Retry with Exponential Backoff

*For any* failed communication with an orchestrator, the system should retry with exponential backoff up to 3 attempts.

**Validates: Requirements 5.5**

### Property 24: Pattern Persistence

*For any* learned attack pattern, it should be successfully persisted to the database or queued for retry.

**Validates: Requirements 6.1**

### Property 25: Pattern Field Completeness

*For any* stored attack pattern, it should include telemetry features, action, reward, and timestamp.

**Validates: Requirements 6.2**

### Property 26: Database Unavailability Handling

*For any* database connection failure, patterns should be queued locally and retried.

**Validates: Requirements 6.3**

### Property 27: Query Result Confidence Scores

*For any* database query for attack patterns, results should include confidence scores.

**Validates: Requirements 6.4**

### Property 28: Efficient Pattern Retrieval

*For any* query by attack type, timestamp, or feature similarity, results should be returned efficiently.

**Validates: Requirements 6.5**

### Property 29: Incremental Learning Without Full Retraining

*For any* new telemetry data, it should be incorporated into learning without triggering full model retraining.

**Validates: Requirements 7.1**

### Property 30: Experience Aggregation Across Instances

*For any* set of active sandbox instances, experiences from all instances should be aggregated.

**Validates: Requirements 7.2**

### Property 31: Model Distribution on Update

*For any* policy network update, the new model should be distributed to all active inference engines.

**Validates: Requirements 7.3**

### Property 32: Performance Degradation Alerting

*For any* detected learning performance degradation, an alert should be triggered.

**Validates: Requirements 7.4**

### Property 33: Complete Logging of Operations

*For any* state transition, action, or reward, a log entry should be created.

**Validates: Requirements 8.1**

### Property 34: Training Metrics Exposure

*For any* training step, metrics including loss, epsilon, and average reward should be exposed.

**Validates: Requirements 8.2**

### Property 35: Inference Logging

*For any* inference operation, the selected action and confidence scores should be logged.

**Validates: Requirements 8.3**

### Property 36: Error Logging with Stack Traces

*For any* error condition, a detailed log entry with stack trace should be created.

**Validates: Requirements 8.4**

### Property 37: Hyperparameter Application

*For any* valid configuration file, the specified hyperparameters should be applied to the DRL agent.

**Validates: Requirements 9.2**

### Property 38: Configuration Hot-Reload

*For any* configuration update, the new values should be applied without requiring a full restart.

**Validates: Requirements 9.4**

### Property 39: Environment-Specific Configuration

*For any* deployment environment (dev/staging/prod), the correct environment-specific configuration should be loaded.

**Validates: Requirements 9.5**

### Property 40: Sandbox1 Learning Integration

*For any* malware detection by sandbox1, the DRL framework should store the experience for learning.

**Validates: Requirements 10.1**

### Property 41: Sandbox2 Learning Integration

*For any* file processed by sandbox2, the DRL framework should store the experience for learning.

**Validates: Requirements 10.2**

### Property 42: Two-Stage Experience Accumulation

*For any* complete two-stage sandbox workflow, experiences from both sandbox1 and sandbox2 should be present in the replay buffer.

**Validates: Requirements 10.3**

### Property 43: Episode Trajectory Completeness

*For any* completed episode (file fully processed), the complete trajectory should be stored in the database.

**Validates: Requirements 10.4**

### Property 44: Graceful Partial Episode Handling

*For any* sandbox failure during processing, the DRL framework should handle the partial episode without crashing.

**Validates: Requirements 10.5**

## Error Handling

### Telemetry Processing Errors
- **Missing Fields**: Fill with configured default values, log warning
- **Malformed Data**: Skip corrupted fields, use defaults, continue processing
- **Buffer Overflow**: Drop oldest entries, log warning, trigger alert if persistent

### Model Loading Errors
- **Corrupted ONNX File**: Fall back to previous valid model, log error
- **Missing Model File**: Use rule-based detection, trigger alert for manual intervention
- **Version Mismatch**: Attempt compatibility mode, log warning

### Communication Errors
- **Sandbox Unreachable**: Retry with exponential backoff (1s, 2s, 4s), max 3 attempts
- **Database Connection Lost**: Queue operations locally, retry every 30 seconds
- **Timeout**: Log timeout, mark operation as failed, continue with next operation

### Training Errors
- **Gradient Explosion**: Clip gradients, reduce learning rate, log warning
- **NaN Loss**: Reset to last checkpoint, reduce learning rate, log error
- **Memory Exhaustion**: Reduce batch size, clear old experiences, log warning

## Testing Strategy

### Unit Testing

**Scope**: Individual components in isolation

**Key Test Areas**:
- Environment Adapter: Test normalization, missing field handling, dimension consistency
- Replay Buffer: Test capacity limits, sampling, concurrent access
- Action Dispatcher: Test retry logic, timeout handling, message formatting
- Database Client: Test CRUD operations, connection handling, query performance

**Framework**: Google Test (C++), pytest (Python)

### Property-Based Testing

**Scope**: Universal properties that should hold across all inputs

**Library**: Hypothesis (Python), RapidCheck (C++)

**Configuration**: Minimum 100 iterations per property test

**Key Properties to Test**:
- Property 2: State vector dimension consistency
- Property 3: Graceful error handling
- Property 6: Experience storage completeness
- Property 17: Argmax action selection
- Property 25: Pattern field completeness

### Integration Testing

**Scope**: Component interactions and data flow

**Key Test Scenarios**:
- End-to-end telemetry flow from sandbox to database
- Model training, export, and loading pipeline
- Action dispatch and feedback loop
- Database persistence and retrieval
- Multi-sandbox coordination

### Performance Testing

**Scope**: Latency, throughput, and resource usage

**Key Metrics**:
- Telemetry processing latency (target: <100ms)
- Inference latency (target: <10ms)
- Training throughput (experiences/second)
- Memory usage under load
- Database query performance

### Stress Testing

**Scope**: System behavior under extreme conditions

**Scenarios**:
- High-frequency telemetry streams (1000+ events/second)
- Large replay buffer (100K+ experiences)
- Concurrent sandbox instances (10+ simultaneous)
- Extended operation (24+ hours continuous)
- Network failures and recovery

## Deployment Considerations

### Python Training Pipeline

**Environment**: Google Colab or dedicated training server
**Dependencies**: PyTorch, NumPy, ONNX, protobuf
**Resources**: GPU recommended for faster training
**Artifacts**: ONNX model files, training logs, metadata JSON

### C++ Inference Engine

**Environment**: Production servers alongside sandbox orchestrators
**Dependencies**: ONNX Runtime, gRPC, PostgreSQL client, spdlog
**Resources**: CPU-only sufficient for inference
**Deployment**: Docker container or system service

### Model Versioning

- Models stored in `models/onnx/drl_agent_v{timestamp}.onnx`
- Metadata stored in `models/onnx/drl_agent_v{timestamp}_metadata.json`
- Symlink `models/onnx/drl_agent_latest.onnx` points to current production model
- Rollback capability by updating symlink

### Monitoring and Observability

**Metrics to Track**:
- Inference latency (p50, p95, p99)
- Training loss and reward trends
- Action distribution
- Database query performance
- Error rates by type

**Logging**:
- Structured JSON logs
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Rotation: Daily, max 30 days retention
- Integration with ELK stack or similar

### Configuration Management

**Config Files**:
- `config/drl_config.json`: Hyperparameters, model paths, endpoints
- `config/db_config.json`: Database connection strings
- Environment variables for sensitive data (passwords, API keys)

**Hot-Reload Support**:
- Watch config files for changes
- Reload without service restart
- Validate before applying
- Log all configuration changes

## Security Considerations

### Model Integrity
- Verify ONNX model checksums before loading
- Sign models with cryptographic signatures
- Detect and prevent model poisoning attacks

### Data Privacy
- Sanitize telemetry data before storage
- Encrypt sensitive patterns in database
- Implement access controls for learned patterns

### Communication Security
- Use TLS for all gRPC connections
- Authenticate sandbox orchestrators
- Rate limit action dispatch to prevent abuse

## Future Enhancements

1. **Multi-Agent Learning**: Support multiple specialized agents for different attack types
2. **Transfer Learning**: Pre-train on public malware datasets, fine-tune on local data
3. **Explainability**: Integrate with XAI module to explain action decisions
4. **Active Learning**: Prioritize uncertain samples for human labeling
5. **Distributed Training**: Scale training across multiple GPUs/nodes
6. **Advanced Algorithms**: Experiment with PPO, SAC, or other state-of-the-art RL algorithms
