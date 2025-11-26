"""
Telemetry Stream - Real telemetry ingestion for DRL training

This module provides real telemetry data ingestion from recorded logs or live streams.
"""

import json
import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path


class TelemetryStream:
    """
    Real telemetry ingestion for DRL training.
    
    Supports:
    - Loading recorded telemetry from JSON files
    - Episodic iteration for training
    - Real-time simulation mode
    
    Args:
        feature_dim: Number of telemetry features
        source: Path to telemetry JSON file or 'live' for streaming
        realtime: If True, simulates live events with delays
        delay: Delay between events in realtime mode (seconds)
    """
    
    def __init__(self, feature_dim: int, source: str = 'telemetry_log.json',
                 realtime: bool = False, delay: float = 0.1):
        self.feature_dim = feature_dim
        self.source = source
        self.realtime = realtime
        self.delay = delay
        self.telemetry_events: Optional[List[Dict[str, Any]]] = None
        self.current_index = 0
        
        # Load telemetry data
        self._load_telemetry()
    
    def _load_telemetry(self) -> None:
        """Load telemetry data from source."""
        if self.source == 'live':
            # Live streaming not implemented yet
            raise NotImplementedError(
                "Live telemetry streaming not yet implemented. "
                "Use recorded telemetry JSON file."
            )
        
        # Load from JSON file
        source_path = Path(self.source)
        if not source_path.exists():
            raise FileNotFoundError(f"Telemetry file not found: {self.source}")
        
        with open(source_path, 'r') as f:
            self.telemetry_events = json.load(f)
        
        if not isinstance(self.telemetry_events, list):
            raise ValueError("Telemetry JSON must be a list of event dictionaries")
        
        print(f"Loaded {len(self.telemetry_events)} telemetry events from {self.source}")
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset stream to beginning for a new episode.
        
        Returns:
            Initial telemetry event
        """
        self.current_index = 0
        
        if not self.telemetry_events:
            raise RuntimeError("No telemetry events loaded")
        
        return self.telemetry_events[self.current_index]
    
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool]:
        """
        Get next telemetry event and compute reward.
        
        Args:
            action: Action taken by agent (used for reward computation)
        
        Returns:
            Tuple of (next_telemetry, reward, done)
        """
        self.current_index += 1
        done = self.current_index >= len(self.telemetry_events) - 1
        
        # Simulate real-time delay if enabled
        if self.realtime:
            time.sleep(self.delay)
        
        # Get current telemetry
        if done:
            telemetry = self.telemetry_events[-1]
        else:
            telemetry = self.telemetry_events[self.current_index]
        
        # Compute reward from telemetry
        reward = self._compute_reward(telemetry, action)
        
        return telemetry, reward, done
    
    def _compute_reward(self, telemetry: Dict[str, Any], action: int) -> float:
        """
        Compute reward based on telemetry and action taken.
        
        Reward structure:
        - Positive reward for correctly identifying malware
        - Negative reward for false positives/negatives
        - Scaled by confidence and severity
        
        Args:
            telemetry: Current telemetry data
            action: Action taken (0=ignore, 1=contain, 2=terminate, 3=quarantine, 4=review)
        
        Returns:
            Reward value
        """
        # Extract ground truth if available
        is_malicious = telemetry.get('is_malicious', False)
        severity = telemetry.get('severity', 0.5)  # 0-1 scale
        
        # Get reward from telemetry if pre-computed
        if 'reward' in telemetry:
            return float(telemetry['reward'])
        
        # Compute reward based on action appropriateness
        # Action mapping: 0=ignore, 1=contain, 2=terminate, 3=quarantine, 4=review
        
        if is_malicious:
            # Malware present - aggressive actions are good
            if action in [2, 3]:  # Terminate or quarantine
                reward = 1.0 * severity
            elif action == 1:  # Contain
                reward = 0.5 * severity
            elif action == 4:  # Review
                reward = 0.2 * severity
            else:  # Ignore - bad!
                reward = -1.0 * severity
        else:
            # Benign file - conservative actions are good
            if action == 0:  # Ignore - correct
                reward = 0.5
            elif action == 4:  # Review - cautious
                reward = 0.3
            elif action == 1:  # Contain - overly cautious
                reward = -0.2
            else:  # Terminate/quarantine - false positive!
                reward = -1.0
        
        return reward
    
    def __len__(self) -> int:
        """Return total number of telemetry events."""
        return len(self.telemetry_events) if self.telemetry_events else 0
    
    def get_episode_length(self) -> int:
        """Return length of current episode."""
        return len(self.telemetry_events) if self.telemetry_events else 0


def create_sample_telemetry_file(output_path: str, num_episodes: int = 10,
                                 steps_per_episode: int = 50) -> None:
    """
    Create a sample telemetry JSON file for testing.
    
    Args:
        output_path: Path to save JSON file
        num_episodes: Number of episodes to generate
        steps_per_episode: Steps per episode
    """
    telemetry_data = []
    
    for episode in range(num_episodes):
        # Randomly decide if this episode contains malware
        is_malicious = np.random.rand() > 0.5
        severity = np.random.rand() if is_malicious else 0.0
        
        for step in range(steps_per_episode):
            # Generate realistic telemetry
            event = {
                'episode_id': episode,
                'step': step,
                'is_malicious': is_malicious,
                'severity': severity,
                
                # System call features
                'syscall_count': int(np.random.exponential(100) * (2 if is_malicious else 1)),
                'syscall_types': ['open', 'read', 'write', 'close'],
                
                # File I/O features
                'file_read_count': int(np.random.poisson(10) * (1.5 if is_malicious else 1)),
                'file_write_count': int(np.random.poisson(5) * (2 if is_malicious else 1)),
                'file_delete_count': int(np.random.poisson(1) * (3 if is_malicious else 1)),
                
                # Network features
                'network_connections': int(np.random.poisson(2) * (3 if is_malicious else 1)),
                'bytes_sent': int(np.random.exponential(1000) * (5 if is_malicious else 1)),
                'bytes_received': int(np.random.exponential(1000) * (2 if is_malicious else 1)),
                
                # Process features
                'child_processes': int(np.random.poisson(1) * (2 if is_malicious else 1)),
                'cpu_usage': float(np.random.rand() * 50 * (1.5 if is_malicious else 1)),
                'memory_usage': float(np.random.exponential(100) * (2 if is_malicious else 1)),
                
                # Behavioral indicators
                'registry_modification': bool(np.random.rand() < (0.7 if is_malicious else 0.1)),
                'privilege_escalation_attempt': bool(np.random.rand() < (0.5 if is_malicious else 0.05)),
                'code_injection_detected': bool(np.random.rand() < (0.6 if is_malicious else 0.02)),
            }
            
            telemetry_data.append(event)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(telemetry_data, f, indent=2)
    
    print(f"Created sample telemetry file: {output_path}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Total events: {len(telemetry_data)}")


if __name__ == "__main__":
    # Create sample telemetry for testing
    create_sample_telemetry_file('sample_telemetry.json', num_episodes=20, steps_per_episode=50)
