"""
DRL Agent - Complete DQN implementation with training

This module implements the full DRL agent with DQN algorithm, including
epsilon-greedy exploration, experience replay, and target network updates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple
from drl_agent_network import DRLAgentNetwork
from replay_buffer import ReplayBuffer


class DRLAgent:
    """
    Deep Q-Network (DQN) agent for reinforcement learning.
    
    Features:
    - Epsilon-greedy exploration
    - Experience replay for stable training
    - Target network for stable Q-value estimation
    - Gradient clipping for training stability
    
    Args:
        input_dim: Dimension of state vector
        output_dim: Number of possible actions
        gamma: Discount factor for future rewards
        lr: Learning rate
        batch_size: Mini-batch size for training
        update_target_steps: Steps between target network updates
        device: torch device ('cuda' or 'cpu')
    """
    
    def __init__(self, input_dim: int, output_dim: int, gamma: float = 0.99,
                 lr: float = 1e-4, batch_size: int = 64, 
                 update_target_steps: int = 1000, device: str = 'cuda'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_steps = update_target_steps
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Policy and target networks
        self.policy_net = DRLAgentNetwork(input_dim, output_dim).to(self.device)
        self.target_net = DRLAgentNetwork(input_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is always in eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
        # Training statistics
        self.steps_done = 0
        self.episodes_done = 0
        self.total_loss = 0.0
        self.loss_count = 0
        
        print(f"DRL Agent initialized on device: {self.device}")
        print(f"  Input dim: {input_dim}, Output dim: {output_dim}")
        print(f"  Gamma: {gamma}, LR: {lr}, Batch size: {batch_size}")
    
    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state vector
            epsilon: Exploration probability
        
        Returns:
            Selected action index
        """
        # Epsilon-greedy exploration
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.output_dim)
        
        # Greedy action from policy network
        self.policy_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
        
        return action
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool) -> None:
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Episode termination flag
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def update(self) -> float:
        """
        Perform one training update using experience replay.
        
        Returns:
            Loss value (0.0 if not enough samples)
        """
        # Check if enough samples
        if not self.replay_buffer.is_ready(self.batch_size):
            return 0.0
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q-values
        self.policy_net.train()
        current_q_values = self.policy_net(states_t).gather(1, actions_t).squeeze()
        
        # Compute target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(dim=1)[0]
            target_q_values = rewards_t + (1 - dones_t) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # Update statistics
        self.steps_done += 1
        self.total_loss += loss.item()
        self.loss_count += 1
        
        # Update target network periodically
        if self.steps_done % self.update_target_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Target network updated at step {self.steps_done}")
        
        return loss.item()
    
    def get_average_loss(self) -> float:
        """Get average loss over recent updates."""
        if self.loss_count == 0:
            return 0.0
        avg_loss = self.total_loss / self.loss_count
        # Reset counters
        self.total_loss = 0.0
        self.loss_count = 0
        return avg_loss
    
    def save_model(self, filepath: str) -> None:
        """
        Save policy network state dict.
        
        Args:
            filepath: Path to save model
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load policy network state dict.
        
        Args:
            filepath: Path to load model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint.get('steps_done', 0)
        self.episodes_done = checkpoint.get('episodes_done', 0)
        print(f"Model loaded from {filepath}")
    
    def export_onnx(self, filepath: str) -> None:
        """
        Export policy network to ONNX format.
        
        Args:
            filepath: Path to save ONNX model
        """
        self.policy_net.eval()
        dummy_input = torch.randn(1, self.input_dim).to(self.device)
        
        torch.onnx.export(
            self.policy_net,
            dummy_input,
            filepath,
            input_names=['state'],
            output_names=['q_values'],
            dynamic_axes={'state': {0: 'batch_size'}},
            opset_version=11,
            do_constant_folding=True
        )
        print(f"Model exported to ONNX: {filepath}")
