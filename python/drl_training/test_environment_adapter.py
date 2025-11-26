"""
Property-based tests for DRLEnvironmentAdapter

Tests universal properties for telemetry processing.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from environment_adapter import DRLEnvironmentAdapter


# Property 2: State Vector Dimension Consistency
# For any raw telemetry input, the Environment Adapter should produce 
# a state vector of exactly the configured feature dimension
@given(
    feature_dim=st.integers(min_value=10, max_value=100),
    syscall_count=st.integers(min_value=0, max_value=10000),
    file_read_count=st.integers(min_value=0, max_value=1000),
    network_connections=st.integers(min_value=0, max_value=100)
)
@settings(max_examples=100)
def test_state_vector_dimension_consistency(feature_dim, syscall_count, 
                                            file_read_count, network_connections):
    """
    **Feature: drl-framework, Property 2: State Vector Dimension Consistency**
    
    For any raw telemetry input, the output state vector should have exactly
    the configured feature dimension.
    """
    adapter = DRLEnvironmentAdapter(feature_dim=feature_dim)
    
    # Create telemetry with random values
    telemetry = {
        'syscall_count': syscall_count,
        'file_read_count': file_read_count,
        'network_connections': network_connections,
        'cpu_usage': np.random.rand() * 100,
        'memory_usage': np.random.rand() * 1000,
    }
    
    state = adapter.process_telemetry(telemetry)
    
    # Verify dimension
    assert state.shape == (feature_dim,)
    assert len(state) == feature_dim
    
    # Verify it's a numpy array of floats
    assert isinstance(state, np.ndarray)
    assert state.dtype == np.float32


# Property 3: Graceful Error Handling for Malformed Telemetry
# For any telemetry data with missing or malformed fields, the Environment Adapter 
# should not crash and should fill missing values with configured defaults
@given(
    feature_dim=st.integers(min_value=10, max_value=50),
    include_syscall=st.booleans(),
    include_file_io=st.booleans(),
    include_network=st.booleans(),
    include_process=st.booleans()
)
@settings(max_examples=100)
def test_graceful_error_handling_malformed_telemetry(feature_dim, include_syscall,
                                                      include_file_io, include_network,
                                                      include_process):
    """
    **Feature: drl-framework, Property 3: Graceful Error Handling for Malformed Telemetry**
    
    For any telemetry with missing fields, the adapter should handle it gracefully
    and produce a valid state vector with defaults.
    """
    adapter = DRLEnvironmentAdapter(feature_dim=feature_dim)
    
    # Create incomplete telemetry
    telemetry = {}
    
    if include_syscall:
        telemetry['syscall_count'] = np.random.randint(0, 1000)
    
    if include_file_io:
        telemetry['file_read_count'] = np.random.randint(0, 100)
        telemetry['file_write_count'] = np.random.randint(0, 100)
    
    if include_network:
        telemetry['network_connections'] = np.random.randint(0, 50)
    
    if include_process:
        telemetry['child_processes'] = np.random.randint(0, 10)
    
    # Should not crash
    try:
        state = adapter.process_telemetry(telemetry)
        
        # Verify valid output
        assert state.shape == (feature_dim,)
        assert not np.any(np.isnan(state))
        assert not np.any(np.isinf(state))
        
        # All values should be in valid range [0, 1] after normalization
        assert np.all(state >= 0.0)
        assert np.all(state <= 1.0)
        
    except Exception as e:
        pytest.fail(f"Adapter crashed on incomplete telemetry: {e}")


@given(
    feature_dim=st.integers(min_value=10, max_value=50)
)
@settings(max_examples=100)
def test_empty_telemetry_handling(feature_dim):
    """
    For any feature dimension, processing empty telemetry should produce
    a valid state vector filled with defaults.
    """
    adapter = DRLEnvironmentAdapter(feature_dim=feature_dim)
    
    # Empty telemetry
    telemetry = {}
    
    state = adapter.process_telemetry(telemetry)
    
    assert state.shape == (feature_dim,)
    assert not np.any(np.isnan(state))
    assert not np.any(np.isinf(state))


@given(
    feature_dim=st.integers(min_value=10, max_value=50),
    num_features=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=100)
def test_normalization_bounds(feature_dim, num_features):
    """
    For any telemetry, all normalized values should be in [0, 1] range.
    """
    adapter = DRLEnvironmentAdapter(feature_dim=feature_dim)
    
    # Create telemetry with random values
    telemetry = {}
    feature_names = ['syscall_count', 'file_read_count', 'file_write_count',
                    'network_connections', 'bytes_sent', 'bytes_received',
                    'child_processes', 'cpu_usage', 'memory_usage']
    
    for i in range(min(num_features, len(feature_names))):
        feature = feature_names[i]
        # Use extreme values to test normalization
        telemetry[feature] = np.random.rand() * 100000
    
    state = adapter.process_telemetry(telemetry)
    
    # All values should be normalized to [0, 1]
    assert np.all(state >= 0.0), f"Found values < 0: {state[state < 0]}"
    assert np.all(state <= 1.0), f"Found values > 1: {state[state > 1]}"


def test_handle_missing_fields():
    """Test that handle_missing_fields fills in defaults."""
    adapter = DRLEnvironmentAdapter(feature_dim=30)
    
    # Incomplete telemetry
    telemetry = {
        'syscall_count': 100,
        'file_read_count': 50,
    }
    
    complete = adapter.handle_missing_fields(telemetry)
    
    # Should have original fields
    assert complete['syscall_count'] == 100
    assert complete['file_read_count'] == 50
    
    # Should have defaults for missing fields
    assert 'file_write_count' in complete
    assert 'network_connections' in complete
    assert complete['file_write_count'] == 0.0


def test_boolean_feature_conversion():
    """Test that boolean features are converted to floats."""
    adapter = DRLEnvironmentAdapter(feature_dim=30)
    
    telemetry = {
        'registry_modification': True,
        'privilege_escalation_attempt': False,
        'code_injection_detected': True,
    }
    
    state = adapter.process_telemetry(telemetry)
    
    # Should not crash and should produce valid state
    assert state.shape == (30,)
    assert not np.any(np.isnan(state))


def test_derived_features():
    """Test that derived features are computed correctly."""
    adapter = DRLEnvironmentAdapter(feature_dim=30)
    
    telemetry = {
        'file_read_count': 100,
        'file_write_count': 50,
        'bytes_sent': 1000000,
        'bytes_received': 500000,
        'child_processes': 5,
    }
    
    state = adapter.process_telemetry(telemetry)
    
    # Should compute derived features without errors
    assert state.shape == (30,)
    assert not np.any(np.isnan(state))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
