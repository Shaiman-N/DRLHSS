#!/usr/bin/env python3
"""
Complete production-grade DQN training pipeline for malware detection
Features: GPU acceleration, checkpointing, ONNX export, comprehensive logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import argparse
from datetime import datetime
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter
import sqlite3

class DQN(nn.Module):
    """Deep Q-Network for malware detection"""
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256, 128]):
        super(DQN, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent with target network"""
    def __init__(self, state_dim, action_dim, lr=0.0001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()

    
    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def train_step(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save_checkpoint(self, filepath):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
    
    def export_onnx(self, filepath):
        self.policy_net.eval()
        dummy_input = torch.randn(1, self.state_dim).to(self.device)
        torch.onnx.export(
            self.policy_net, dummy_input, filepath,
            input_names=['state'], output_names=['q_values'],
            dynamic_axes={'state': {0: 'batch_size'}, 'q_values': {0: 'batch_size'}}
        )
        print(f"Model exported to ONNX: {filepath}")

class MalwareEnvironment:
    """Simulated malware detection environment"""
    def __init__(self, state_dim=16, action_dim=4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reset()
    
    def reset(self):
        self.state = np.random.rand(self.state_dim).astype(np.float32)
        self.steps = 0
        return self.state
    
    def step(self, action):
        # Simulate environment dynamics
        self.steps += 1
        
        # Reward based on action correctness (simplified)
        malicious_score = self.state[0]  # First feature indicates maliciousness
        
        if action == 0:  # Allow
            reward = 1.0 if malicious_score < 0.5 else -1.0
        elif action == 1:  # Block
            reward = 1.0 if malicious_score >= 0.5 else -0.5
        elif action == 2:  # Quarantine
            reward = 0.8 if malicious_score >= 0.7 else -0.3
        else:  # Deep scan
            reward = 0.5 if malicious_score >= 0.3 else -0.2
        
        # Next state
        self.state = np.random.rand(self.state_dim).astype(np.float32)
        done = self.steps >= 100
        
        return self.state, reward, done

def train(args):
    """Main training loop"""
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
    
    env = MalwareEnvironment(state_dim=args.state_dim, action_dim=args.action_dim)
    agent = DQNAgent(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        lr=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay
    )
    
    episode_rewards = []
    best_avg_reward = -float('inf')
    
    print(f"Starting training for {args.num_episodes} episodes...")
    
    for episode in range(args.num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = []
        
        while True:
            action = agent.select_action(state, training=True)
            next_state, reward, done = env.step(action)
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            loss = agent.train_step(batch_size=args.batch_size)
            if loss is not None:
                episode_loss.append(loss)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        agent.decay_epsilon()
        
        if episode % args.target_update_freq == 0:
            agent.update_target_network()
        
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:])
        
        writer.add_scalar('Reward/Episode', episode_reward, episode)
        writer.add_scalar('Reward/Average', avg_reward, episode)
        writer.add_scalar('Epsilon', agent.epsilon, episode)
        if episode_loss:
            writer.add_scalar('Loss/Average', np.mean(episode_loss), episode)
        
        if episode % args.log_interval == 0:
            print(f"Episode {episode}/{args.num_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
        
        if episode % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_{episode}.pt')
            agent.save_checkpoint(checkpoint_path)
        
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_model_path = os.path.join(args.output_dir, 'best_model.pt')
            agent.save_checkpoint(best_model_path)
    
    # Export final model to ONNX
    onnx_path = os.path.join(args.output_dir, 'dqn_model.onnx')
    agent.export_onnx(onnx_path)
    
    # Save metadata
    metadata = {
        'model_version': '1.0.0',
        'training_date': datetime.now().isoformat(),
        'training_episodes': args.num_episodes,
        'final_average_reward': float(avg_reward),
        'learning_rate': args.learning_rate,
        'gamma': args.gamma,
        'epsilon_start': args.epsilon_start,
        'epsilon_end': args.epsilon_end,
        'batch_size': args.batch_size,
        'target_update_frequency': args.target_update_freq,
        'input_dim': args.state_dim,
        'output_dim': args.action_dim,
        'hidden_layers': [256, 256, 128]
    }
    
    metadata_path = os.path.join(args.output_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    writer.close()
    print(f"Training complete! Best average reward: {best_avg_reward:.2f}")
    print(f"Model saved to: {onnx_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DQN for malware detection')
    parser.add_argument('--state-dim', type=int, default=16, help='State dimension')
    parser.add_argument('--action-dim', type=int, default=4, help='Action dimension')
    parser.add_argument('--num-episodes', type=int, default=10000, help='Number of episodes')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='Initial epsilon')
    parser.add_argument('--epsilon-end', type=float, default=0.1, help='Final epsilon')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Epsilon decay')
    parser.add_argument('--target-update-freq', type=int, default=100, help='Target network update frequency')
    parser.add_argument('--checkpoint-interval', type=int, default=1000, help='Checkpoint save interval')
    parser.add_argument('--log-interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    
    args = parser.parse_args()
    train(args)
