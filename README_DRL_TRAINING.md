# DRL Framework - Training Pipeline

## 🎉 What's Ready

You now have a **complete, production-grade DRL training pipeline** that can:

- ✅ Load real telemetry data from your sandboxes
- ✅ Train using Deep Q-Network (DQN) algorithm
- ✅ Utilize your RTX 4050 GPU automatically
- ✅ Export trained models to ONNX format
- ✅ Run comprehensive property-based tests
- ✅ Generate training statistics and checkpoints

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd python/drl_training
pip install -r requirements.txt
```

### 2. Train Your First Model

```bash
python train_drl.py
```

This will:
- Create sample telemetry data (if needed)
- Train for 500 episodes
- Use your GPU automatically
- Save checkpoints every 50 episodes
- Export final model to ONNX
- Generate metadata JSON

### 3. Verify Output

Check the `models/` directory for:
- `drl_agent_final_YYYYMMDD_HHMMSS.pth` - PyTorch model
- `drl_agent_YYYYMMDD_HHMMSS.onnx` - ONNX model (for C++)
- `drl_agent_YYYYMMDD_HHMMSS_metadata.json` - Training info

## 📊 Training Configuration

Default settings (in `train_drl.py`):
- **Episodes:** 500
- **Feature Dimension:** 30 (telemetry features)
- **Actions:** 5 (ignore, contain, terminate, quarantine, review)
- **Learning Rate:** 0.0001
- **Gamma (discount):** 0.99
- **Batch Size:** 64
- **Epsilon:** 1.0 → 0.1 (exploration decay)

## 🧪 Run Tests

```bash
# Test replay buffer
pytest test_replay_buffer.py -v

# Test environment adapter
pytest test_environment_adapter.py -v

# Run all tests
pytest -v
```

## 📁 File Structure

```
python/drl_training/
├── drl_agent_network.py      # Neural network architecture
├── replay_buffer.py           # Experience replay
├── environment_adapter.py     # Telemetry normalization
├── telemetry_stream.py        # Data loading
├── drl_agent.py              # Complete DQN agent
├── train_drl.py              # Training script
├── test_replay_buffer.py     # Property tests
├── test_environment_adapter.py # Property tests
└── requirements.txt          # Dependencies
```

## 🎯 Using Your Own Telemetry

### Format

Create a JSON file with telemetry events:

```json
[
  {
    "episode_id": 0,
    "step": 0,
    "is_malicious": true,
    "severity": 0.8,
    "syscall_count": 150,
    "file_read_count": 20,
    "file_write_count": 10,
    "network_connections": 5,
    "bytes_sent": 5000,
    "bytes_received": 2000,
    "child_processes": 2,
    "cpu_usage": 45.5,
    "memory_usage": 250.0,
    "registry_modification": true,
    "privilege_escalation_attempt": false,
    "code_injection_detected": true
  },
  ...
]
```

### Train with Custom Data

```python
from train_drl import train_drl_agent

agent, rewards, losses = train_drl_agent(
    telemetry_file="my_telemetry.json",
    num_episodes=1000,
    feature_dim=30,
    action_dim=5
)
```

## 🔧 Customization

### Adjust Network Architecture

Edit `drl_agent_network.py`:

```python
self.network = nn.Sequential(
    nn.Linear(input_dim, 512),  # Increase neurons
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, output_dim)
)
```

### Change Hyperparameters

Edit `train_drl.py` or pass to `train_drl_agent()`:

```python
agent = DRLAgent(
    input_dim=feature_dim,
    output_dim=action_dim,
    gamma=0.95,              # Adjust discount
    lr=5e-4,                 # Adjust learning rate
    batch_size=128,          # Larger batches
    update_target_steps=500  # More frequent updates
)
```

## 📈 Monitoring Training

The training script shows:
- Episode progress bar
- Average reward (last 10 episodes)
- Average loss (last 10 episodes)
- Current epsilon (exploration rate)
- Replay buffer size

Example output:
```
Episode 100/500
  Avg Reward (last 10): 12.45
  Avg Loss (last 10): 0.0234
  Epsilon: 0.850
  Buffer size: 5000
```

## 🎓 Understanding the Output

### Rewards
- **Positive:** Good decisions (correctly identifying malware)
- **Negative:** Bad decisions (false positives/negatives)
- **Trend:** Should increase over training

### Loss
- **MSE Loss:** Difference between predicted and target Q-values
- **Trend:** Should decrease and stabilize

### Epsilon
- **Starts at 1.0:** Pure exploration
- **Decays to 0.1:** Mostly exploitation
- **Balances:** Exploration vs learned policy

## 🐛 Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```python
agent = DRLAgent(..., batch_size=32)
```

### Training Too Slow
- Ensure GPU is being used (check output)
- Reduce number of episodes
- Use smaller network

### Poor Performance
- Increase training episodes
- Adjust reward function in `telemetry_stream.py`
- Tune hyperparameters
- Check telemetry data quality

## 🔜 Next Steps

1. **Train a model** with the provided script
2. **Verify ONNX export** works
3. **Tomorrow:** Implement C++ inference engine
4. **Then:** Integrate with your sandboxes

## 📚 Key Concepts

### DQN (Deep Q-Network)
- Learns Q-values: Q(state, action) = expected future reward
- Uses neural network to approximate Q-function
- Selects action with highest Q-value

### Experience Replay
- Stores past experiences in buffer
- Samples random batches for training
- Breaks temporal correlations
- Improves stability

### Target Network
- Separate network for computing target Q-values
- Updated periodically (every 1000 steps)
- Prevents moving target problem
- Stabilizes training

### Epsilon-Greedy
- With probability ε: explore (random action)
- With probability 1-ε: exploit (best action)
- ε decays over time: exploration → exploitation

## ✅ Quality Assurance

All code includes:
- Comprehensive error handling
- Input validation
- Type hints
- Docstrings
- Property-based tests (100+ iterations)
- GPU/CPU compatibility
- Progress tracking
- Checkpoint saving

## 🎯 Success Criteria

You'll know training is working when:
- ✅ GPU is detected and used
- ✅ Rewards increase over episodes
- ✅ Loss decreases and stabilizes
- ✅ ONNX file is created
- ✅ Metadata JSON is generated
- ✅ Model can be loaded and used

---

**Ready to train your first DRL model!** 🚀

Run `python train_drl.py` and watch your agent learn to detect malware.
