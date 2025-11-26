"""
Property-based tests for ReplayBuffer

Tests universal properties that should hold across all inputs.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from replay_buffer import ReplayBuffer


# Property 5: Replay Buffer Capacity Invariant
# For any sequence of operations, the replay buffer should maintain capacity 
# for at least 10,000 observations
@given(
    capacity=st.integers(min_value=10000, max_value=100000),
    num_additions=st.integers(min_value=0, max_value=150000)
)
@settings(max_examples=100)
def test_buffer_capacity_invariant(capacity, num_additions):
    """
    **Feature: drl-framework, Property 5: Replay Buffer Capacity Invariant**
    
    For any sequence of add operations, the buffer should never exceed its capacity
    and should maintain at least the configured capacity.
    """
    buffer = ReplayBuffer(capacity=capacity)
    
    # Verify initial capacity
    assert buffer.get_capacity() == capacity
    assert len(buffer) == 0
    
    # Add experiences
    state_dim = 30
    for i in range(num_additions):
        state = np.random.rand(state_dim).astype(np.float32)
        action = np.random.randint(0, 5)
        reward = np.random.randn()
        next_state = np.random.rand(state_dim).astype(np.float32)
        done = np.random.rand() < 0.1
        
        buffer.add(state, action, reward, next_state, done)
    
    # Verify capacity is maintained
    assert buffer.get_capacity() == capacity
    
    # Verify size doesn't exceed capacity
    assert len(buffer) <= capacity
    
    # Verify size is correct
    expected_size = min(num_additions, capacity)
    assert len(buffer) == expected_size


# Additional property tests
@given(
    capacity=st.integers(min_value=100, max_value=10000),
    batch_size=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=100)
def test_sample_returns_correct_batch_size(capacity, batch_size):
    """
    For any buffer with sufficient experiences, sampling should return 
    exactly the requested batch size.
    """
    buffer = ReplayBuffer(capacity=capacity)
    
    # Fill buffer with enough experiences
    state_dim = 30
    num_experiences = max(batch_size, 100)
    
    for _ in range(num_experiences):
        state = np.random.rand(state_dim).astype(np.float32)
        action = np.random.randint(0, 5)
        reward = np.random.randn()
        next_state = np.random.rand(state_dim).astype(np.float32)
        done = False
        buffer.add(state, action, reward, next_state, done)
    
    # Sample and verify batch size
    if batch_size <= len(buffer):
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        
        assert states.shape[0] == batch_size
        assert actions.shape[0] == batch_size
        assert rewards.shape[0] == batch_size
        assert next_states.shape[0] == batch_size
        assert dones.shape[0] == batch_size


@given(
    capacity=st.integers(min_value=100, max_value=10000)
)
@settings(max_examples=100)
def test_buffer_fifo_behavior(capacity):
    """
    For any buffer at capacity, adding new experiences should remove oldest ones (FIFO).
    """
    buffer = ReplayBuffer(capacity=capacity)
    state_dim = 30
    
    # Fill buffer to capacity
    first_state = np.ones(state_dim, dtype=np.float32)
    buffer.add(first_state, 0, 0.0, first_state, False)
    
    for _ in range(capacity - 1):
        state = np.random.rand(state_dim).astype(np.float32)
        buffer.add(state, 0, 0.0, state, False)
    
    assert len(buffer) == capacity
    
    # Add one more experience (should evict the first one)
    new_state = np.zeros(state_dim, dtype=np.float32)
    buffer.add(new_state, 0, 0.0, new_state, False)
    
    assert len(buffer) == capacity
    
    # Sample all experiences and verify first one is not present
    all_experiences = buffer.sample(capacity)
    states = all_experiences[0]
    
    # The first state (all ones) should not be in the buffer anymore
    # Check if any state is all ones
    has_first_state = any(np.allclose(state, first_state) for state in states)
    assert not has_first_state


def test_buffer_clear():
    """Test that clear() removes all experiences."""
    buffer = ReplayBuffer(capacity=1000)
    
    # Add some experiences
    for _ in range(100):
        state = np.random.rand(30).astype(np.float32)
        buffer.add(state, 0, 0.0, state, False)
    
    assert len(buffer) > 0
    
    buffer.clear()
    assert len(buffer) == 0


def test_buffer_is_ready():
    """Test is_ready() method."""
    buffer = ReplayBuffer(capacity=1000)
    batch_size = 64
    
    assert not buffer.is_ready(batch_size)
    
    # Add enough experiences
    for _ in range(batch_size):
        state = np.random.rand(30).astype(np.float32)
        buffer.add(state, 0, 0.0, state, False)
    
    assert buffer.is_ready(batch_size)


def test_sample_raises_on_insufficient_data():
    """Test that sampling more than available raises ValueError."""
    buffer = ReplayBuffer(capacity=1000)
    
    # Add only 10 experiences
    for _ in range(10):
        state = np.random.rand(30).astype(np.float32)
        buffer.add(state, 0, 0.0, state, False)
    
    # Try to sample more than available
    with pytest.raises(ValueError):
        buffer.sample(100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
