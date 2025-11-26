"""
Environment Adapter - Converts raw telemetry to state vectors

This module normalizes telemetry data into fixed-dimension state vectors
suitable for neural network input.
"""

import numpy as np
from typing import Dict, Any, List


class DRLEnvironmentAdapter:
    """
    Adapter for converting raw telemetry data to normalized state vectors.
    
    Handles missing fields, normalization, and maintains consistent feature ordering.
    
    Args:
        feature_dim: Dimension of output state vector
        feature_names: Ordered list of feature names to extract
    """
    
    def __init__(self, feature_dim: int, feature_names: List[str] = None):
        self.feature_dim = feature_dim
        
        # Default feature names if not provided
        if feature_names is None:
            self.feature_names = self._get_default_feature_names()
        else:
            self.feature_names = feature_names
        
        # Default values for missing fields
        self.default_values = self._initialize_defaults()
        
        # Normalization parameters (can be learned from data)
        self.normalization_params = self._initialize_normalization()
    
    def _get_default_feature_names(self) -> List[str]:
        """Get default ordered list of telemetry feature names."""
        return [
            'syscall_count',
            'file_read_count',
            'file_write_count',
            'file_delete_count',
            'network_connections',
            'bytes_sent',
            'bytes_received',
            'child_processes',
            'cpu_usage',
            'memory_usage',
            'registry_modification',
            'privilege_escalation_attempt',
            'code_injection_detected',
            # Additional derived features
            'file_io_ratio',
            'network_intensity',
            'process_activity',
        ]
    
    def _initialize_defaults(self) -> Dict[str, float]:
        """Initialize default values for missing fields."""
        return {
            'syscall_count': 0.0,
            'file_read_count': 0.0,
            'file_write_count': 0.0,
            'file_delete_count': 0.0,
            'network_connections': 0.0,
            'bytes_sent': 0.0,
            'bytes_received': 0.0,
            'child_processes': 0.0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'registry_modification': 0.0,
            'privilege_escalation_attempt': 0.0,
            'code_injection_detected': 0.0,
            'file_io_ratio': 0.0,
            'network_intensity': 0.0,
            'process_activity': 0.0,
        }
    
    def _initialize_normalization(self) -> Dict[str, Dict[str, float]]:
        """Initialize normalization parameters (min, max, mean, std)."""
        # These should ideally be learned from training data
        # Using reasonable defaults for now
        return {
            'syscall_count': {'min': 0, 'max': 10000, 'mean': 1000, 'std': 2000},
            'file_read_count': {'min': 0, 'max': 1000, 'mean': 50, 'std': 100},
            'file_write_count': {'min': 0, 'max': 1000, 'mean': 50, 'std': 100},
            'file_delete_count': {'min': 0, 'max': 100, 'mean': 5, 'std': 10},
            'network_connections': {'min': 0, 'max': 100, 'mean': 10, 'std': 20},
            'bytes_sent': {'min': 0, 'max': 1000000, 'mean': 10000, 'std': 50000},
            'bytes_received': {'min': 0, 'max': 1000000, 'mean': 10000, 'std': 50000},
            'child_processes': {'min': 0, 'max': 50, 'mean': 2, 'std': 5},
            'cpu_usage': {'min': 0, 'max': 100, 'mean': 30, 'std': 20},
            'memory_usage': {'min': 0, 'max': 16000, 'mean': 500, 'std': 1000},
        }
    
    def process_telemetry(self, telemetry: Dict[str, Any]) -> np.ndarray:
        """
        Convert raw telemetry dictionary to normalized state vector.
        
        Args:
            telemetry: Dictionary containing telemetry data
        
        Returns:
            Normalized state vector of shape (feature_dim,)
        """
        state = np.zeros(self.feature_dim, dtype=np.float32)
        
        # Extract and normalize features
        for i, feature_name in enumerate(self.feature_names[:self.feature_dim]):
            if feature_name in telemetry:
                raw_value = telemetry[feature_name]
                # Convert boolean to float
                if isinstance(raw_value, bool):
                    raw_value = float(raw_value)
                # Normalize
                normalized_value = self._normalize_feature(feature_name, raw_value)
                state[i] = normalized_value
            else:
                # Use default value
                state[i] = self.default_values.get(feature_name, 0.0)
        
        # Compute derived features if needed
        state = self._add_derived_features(state, telemetry)
        
        return state
    
    def _normalize_feature(self, feature_name: str, value: float) -> float:
        """
        Normalize a feature value to [0, 1] range.
        
        Args:
            feature_name: Name of the feature
            value: Raw feature value
        
        Returns:
            Normalized value in [0, 1]
        """
        if feature_name in self.normalization_params:
            params = self.normalization_params[feature_name]
            min_val = params['min']
            max_val = params['max']
            
            # Min-max normalization
            if max_val > min_val:
                normalized = (value - min_val) / (max_val - min_val)
                # Clip to [0, 1]
                normalized = np.clip(normalized, 0.0, 1.0)
                return normalized
        
        # Default: clip to [0, 1]
        return np.clip(value, 0.0, 1.0)
    
    def _add_derived_features(self, state: np.ndarray, 
                             telemetry: Dict[str, Any]) -> np.ndarray:
        """
        Add derived features computed from raw telemetry.
        
        Args:
            state: Current state vector
            telemetry: Raw telemetry dictionary
        
        Returns:
            State vector with derived features
        """
        # Compute file I/O ratio
        file_reads = telemetry.get('file_read_count', 0)
        file_writes = telemetry.get('file_write_count', 0)
        if file_reads + file_writes > 0:
            file_io_ratio = file_writes / (file_reads + file_writes)
        else:
            file_io_ratio = 0.0
        
        # Compute network intensity
        bytes_sent = telemetry.get('bytes_sent', 0)
        bytes_received = telemetry.get('bytes_received', 0)
        network_intensity = (bytes_sent + bytes_received) / 1000000.0  # Normalize to MB
        network_intensity = np.clip(network_intensity, 0.0, 1.0)
        
        # Compute process activity
        child_procs = telemetry.get('child_processes', 0)
        process_activity = np.clip(child_procs / 10.0, 0.0, 1.0)
        
        # Add to state if there's room
        if len(self.feature_names) > len(state):
            # Extend state with derived features
            derived = np.array([file_io_ratio, network_intensity, process_activity], 
                             dtype=np.float32)
            state = np.concatenate([state, derived])
        
        return state[:self.feature_dim]  # Ensure correct dimension
    
    def handle_missing_fields(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fill in missing fields with default values.
        
        Args:
            telemetry: Telemetry dictionary (possibly incomplete)
        
        Returns:
            Complete telemetry dictionary with defaults filled in
        """
        complete_telemetry = telemetry.copy()
        
        for feature_name in self.feature_names:
            if feature_name not in complete_telemetry:
                complete_telemetry[feature_name] = self.default_values.get(feature_name, 0.0)
        
        return complete_telemetry
