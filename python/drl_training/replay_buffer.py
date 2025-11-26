"""
Replay Buffer - Experience storage for stable DRL training

This module implements an experience replay buffer to break temporal correlations
in training data and improve learning stability.
"""

import random
from collections import deque
from typing import List, Tuple
import numpy as np


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling training experiences.
    
    Uses a deque with fixed capacity for efficient FIFO operations.
    
    Args:
        capacity: Maximum number of experiences to store
    """
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool) -> None:
        """
        Add an experience tuple to the buffer.
        
        Args:
            state: Current state vector
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode terminated
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                 np.ndarray, np.ndarray]:
        """
        Sample a random batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as numpy arrays
        
        Raises:
            ValueError: If batch_size > buffer size
        """
        if batch_size > len(self.buffer):
            raise ValueError(f"Cannot sample {batch_size} experiences from buffer "
                           f"with only {len(self.buffer)} experiences")
        
        experiences = random.sample(self.buffer, batch_size)
        
        states = np.array([exp[0] for exp in experiences], dtype=np.float32)
        actions = np.array([exp[1] for exp in experiences], dtype=np.int64)
        rewards = np.array([exp[2] for exp in experiences], dtype=np.float32)
        next_states = np.array([exp[3] for exp in experiences], dtype=np.float32)
        dones = np.array([exp[4] for exp in experiences], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current number of experiences in buffer."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """
        Check if buffer has enough experiences for sampling.
        
        Args:
            batch_size: Required batch size
        
        Returns:
            True if buffer size >= batch_size
        """
        return len(self.buffer) >= batch_size
    
    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self.buffer.clear()
    
    def get_capacity(self) -> int:
        """Return the maximum capacity of the buffer."""
        return self.capacity
