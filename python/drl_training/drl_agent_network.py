"""
DRL Agent Network - Neural network for Q-value estimation

This module implements the policy network architecture for the DRL agent.
"""

import torch
import torch.nn as nn


class DRLAgentNetwork(nn.Module):
    """
    Deep Q-Network for estimating Q-values.
    
    Architecture:
        Input Layer -> Hidden(256, ReLU) -> Hidden(256, ReLU) -> Output Layer
    
    Args:
        input_dim: Dimension of state vector (telemetry features)
        output_dim: Number of possible actions
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super(DRLAgentNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier uniform initialization."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor of shape (batch_size, input_dim)
        
        Returns:
            Q-values tensor of shape (batch_size, output_dim)
        """
        return self.network(x)
    
    def get_action(self, state: torch.Tensor) -> int:
        """
        Get the best action for a given state (greedy policy).
        
        Args:
            state: State tensor of shape (1, input_dim) or (input_dim,)
        
        Returns:
            Action index with highest Q-value
        """
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            q_values = self.forward(state)
            return torch.argmax(q_values, dim=1).item()
