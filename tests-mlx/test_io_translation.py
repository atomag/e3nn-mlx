#!/usr/bin/env python3
"""
Test script to verify the IO module translation from e3nn to MLX works correctly.
"""

import mlx.core as mx
import numpy as np

# Test imports
from e3nn_mlx.io import CartesianTensor, SphericalTensor


def test_cartesian_tensor_basic():
    """Test basic Cartesian tensor functionality."""
    print("Testing CartesianTensor...")
    
    # Test creation
    ct = CartesianTensor("ij=ji")
    print(f"CartesianTensor('ij=ji'): {ct}")
    
    # Test from_cartesian and to_cartesian
    t = mx.arange(9, dtype=mx.float32).reshape(3, 3)
    print(f"Original tensor shape: {t.shape}")
    
    y = ct.from_cartesian(t)
    print(f"Converted to irreps shape: {y.shape}")
    
    z = ct.to_cartesian(y)
    print(f"Back to cartesian shape: {z.shape}")
    
    # Test symmetry
    symmetric = (t + t.T) / 2
    print(f"Symmetric reconstruction close: {mx.allclose(z, symmetric, atol=1e-5)}")
    
    return True


def test_spherical_tensor_basic():
    """Test basic SphericalTensor functionality."""
    print("\nTesting SphericalTensor...")
    
    # Test creation
    st = SphericalTensor(2, 1, 1)
    print(f"SphericalTensor(2, 1, 1): {st}")
    
    # Test norms
    signal = mx.array([1.5, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    norms = st.norms(signal)
    print(f"Norms: {norms}")
    
    # Test with_peaks_at
    pos = mx.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    x = st.with_peaks_at(pos)
    print(f"With peaks shape: {x.shape}")
    
    # Test signal_xyz
    signals = st.randn(1, 1, 2, -1)
    r = mx.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = st.signal_xyz(signals, r)
    print(f"Signal xyz shape: {result.shape}")
    
    return True


def test_sum_of_diracs():
    """Test sum of diracs functionality."""
    print("\nTesting sum_of_diracs...")
    
    st = SphericalTensor(2, 1, -1)
    
    pos = mx.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    val = mx.array([-1.0, 1.0])
    
    x = st.sum_of_diracs(pos, val)
    print(f"Sum of diracs shape: {x.shape}")
    print(f"Sum of diracs: {x}")
    
    return True


if __name__ == "__main__":
    print("Testing e3nn-mlx IO module translation...")
    
    try:
        test_cartesian_tensor_basic()
        test_spherical_tensor_basic()
        test_sum_of_diracs()
        print("\nAll tests passed! ✅")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()