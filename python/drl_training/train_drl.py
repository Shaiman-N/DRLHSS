"""
DRL Training Script - Complete training pipeline

This script trains the DRL agent on telemetry data and exports to ONNX.
Can be run locally or on Google Colab.
"""

import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

from drl_agent import DRLAgent
from environment_adapter import DRLEnvironmentAdapter
from telemetry_stream import TelemetryStream, create_sample_telemetry_file


def train_drl_agent(
    telemetry_file: str,
    num_episodes: int = 1000,
    feature_dim: int = 30,
    action_dim: int = 5,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.1,
    epsilon_decay: int = 5000,
    save_interval: int = 100,
    output_dir: str = 'models'
):
    """
    Train DRL agent on telemetry data.
    
    Args:
        telemetry_file: Path to telemetry JSON file
        num_episodes: Number of training episodes
        feature_dim: Dimension of state vector
        action_dim: Number of actions
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Steps for epsilon decay
        save_interval: Episodes between model saves
        output_dir: Directory to save models
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    print("Initializing DRL training...")
    env_adapter = DRLEnvironmentAdapter(feature_dim=feature_dim)
    telemetry_env = TelemetryStream(feature_dim=feature_dim, source=telemetry_file)
    
    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    agent = DRLAgent(
        input_dim=feature_dim,
        output_dim=action_dim,
        gamma=0.99,
        lr=1e-4,
        batch_size=64,
        update_target_steps=1000,
        device=device
    )
    
    # Training statistics
    episode_rewards = []
    episode_losses = []
    epsilon = epsilon_start
    
    print(f"\nStarting training for {num_episodes} episodes...")
    print(f"Epsilon decay: {epsilon_start} -> {epsilon_end} over {epsilon_decay} steps\n")
    
    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training"):
        # Reset environment
        telemetry = telemetry_env.reset()
        state = env_adapter.process_telemetry(telemetry)
        
        episode_reward = 0.0
        episode_loss = 0.0
        loss_count = 0
        done = False
        
        # Episode loop
        while not done:
            # Select action
            action = agent.select_action(state, epsilon)
            
            # Take step
            next_telemetry, reward, done = telemetry_env.step(action)
            next_state = env_adapter.process_telemetry(next_telemetry)
            
            # Store experience
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train
            loss = agent.update()
            if loss > 0:
                episode_loss += loss
                loss_count += 1
            
            # Update state
            state = next_state
            episode_reward += reward
            
            # Decay epsilon
            epsilon = max(epsilon_end, epsilon * np.exp(-1.0 / epsilon_decay))
        
        # Record statistics
        episode_rewards.append(episode_reward)
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0.0
        episode_losses.append(avg_loss)
        agent.episodes_done += 1
        
        # Print progress
        if (episode + 1) % 10 == 0:
            recent_reward = np.mean(episode_rewards[-10:])
            recent_loss = np.mean(episode_losses[-10:])
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward (last 10): {recent_reward:.2f}")
            print(f"  Avg Loss (last 10): {recent_loss:.4f}")
            print(f"  Epsilon: {epsilon:.3f}")
            print(f"  Buffer size: {len(agent.replay_buffer)}")
        
        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            checkpoint_path = output_path / f"drl_agent_episode_{episode + 1}.pth"
            agent.save_model(str(checkpoint_path))
    
    # Training complete
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    # Final statistics
    final_avg_reward = np.mean(episode_rewards[-100:])
    final_avg_loss = np.mean(episode_losses[-100:])
    
    print(f"\nFinal Statistics:")
    print(f"  Average Reward (last 100): {final_avg_reward:.2f}")
    print(f"  Average Loss (last 100): {final_avg_loss:.4f}")
    print(f"  Total Steps: {agent.steps_done}")
    print(f"  Buffer Size: {len(agent.replay_buffer)}")
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = output_path / f"drl_agent_final_{timestamp}.pth"
    agent.save_model(str(final_model_path))
    
    # Export to ONNX
    onnx_path = output_path / f"drl_agent_{timestamp}.onnx"
    agent.export_onnx(str(onnx_path))
    
    # Save metadata
    metadata = {
        'model_version': timestamp,
        'training_date': datetime.now().isoformat(),
        'training_episodes': num_episodes,
        'final_average_reward': float(final_avg_reward),
        'final_loss': float(final_avg_loss),
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'epsilon_start': epsilon_start,
        'epsilon_end': epsilon_end,
        'batch_size': 64,
        'target_update_frequency': 1000,
        'input_dim': feature_dim,
        'output_dim': action_dim,
        'hidden_layers': [256, 256],
        'device': device,
    }
    
    metadata_path = output_path / f"drl_agent_{timestamp}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModel files saved:")
    print(f"  PyTorch: {final_model_path}")
    print(f"  ONNX: {onnx_path}")
    print(f"  Metadata: {metadata_path}")
    
    return agent, episode_rewards, episode_losses


if __name__ == "__main__":
    # Check if sample telemetry exists, create if not
    telemetry_file = "sample_telemetry.json"
    if not Path(telemetry_file).exists():
        print("Creating sample telemetry data...")
        create_sample_telemetry_file(telemetry_file, num_episodes=50, steps_per_episode=50)
    
    # Train agent
    agent, rewards, losses = train_drl_agent(
        telemetry_file=telemetry_file,
        num_episodes=500,  # Adjust based on your needs
        feature_dim=30,
        action_dim=5,
        save_interval=50
    )
    
    print("\nTraining pipeline complete!")
    print("You can now use the ONNX model in your C++ inference engine.")
