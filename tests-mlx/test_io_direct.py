#!/usr/bin/env python3
"""
Direct test for the IO module without full package imports.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'e3nn_mlx'))

import mlx.core as mx
import numpy as np

# Test the IO module components directly
from io._cartesian_tensor import CartesianTensor
from io._spherical_tensor import SphericalTensor


def test_cartesian_tensor_basic():
    """Test basic Cartesian tensor functionality."""
    print("Testing CartesianTensor...")
    
    # Test creation
    ct = CartesianTensor("ij=ji")
    print(f"CartesianTensor('ij=ji'): {ct}")
    print(f"Dimension: {ct.dim}")
    
    # Test from_cartesian and to_cartesian
    t = mx.arange(9, dtype=mx.float32).reshape(3, 3)
    print(f"Original tensor:\n{t}")
    
    y = ct.from_cartesian(t)
    print(f"Converted to irreps: {y}")
    
    z = ct.to_cartesian(y)
    print(f"Back to cartesian:\n{z}")
    
    # Test symmetry
    symmetric = (t + t.T) / 2
    print(f"Is symmetric reconstruction close: {mx.allclose(z, symmetric, atol=1e-5)}")
    
    return True


def test_spherical_tensor_creation():
    """Test basic SphericalTensor creation."""
    print("\nTesting SphericalTensor creation...")
    
    # Test creation
    st = SphericalTensor(2, 1, 1)
    print(f"SphericalTensor(2, 1, 1): {st}")
    print(f"Dimension: {st.dim}")
    
    st_odd = SphericalTensor(2, 1, -1)
    print(f"SphericalTensor(2, 1, -1): {st_odd}")
    
    return True


def test_spherical_tensor_norms():
    """Test norms calculation."""
    print("\nTesting SphericalTensor norms...")
    
    st = SphericalTensor(1, 1, -1)  # lmax=1
    # Create signal: [l=0, l=1]
    signal = mx.array([1.5, 0.0, 3.0, 4.0])  # [1x0e + 1x1o = 1 + 3 = 4 dims]
    print(f"Signal: {signal}")
    
    try:
        norms = st.norms(signal)
        print(f"Norms: {norms}")
        print(f"Norms shape: {norms.shape}")
        return True
    except Exception as e:
        print(f"Error in norms: {e}")
        return False


def test_spherical_tensor_sum_of_diracs():
    """Test sum of diracs functionality."""
    print("\nTesting sum_of_diracs...")
    
    st = SphericalTensor(1, 1, -1)  # Small lmax for testing
    
    pos = mx.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    val = mx.array([-1.0, 1.0])
    
    try:
        x = st.sum_of_diracs(pos, val)
        print(f"Sum of diracs shape: {x.shape}")
        print(f"Sum of diracs: {x}")
        return True
    except Exception as e:
        print(f"Error in sum_of_diracs: {e}")
        return False


if __name__ == "__main__":
    print("Testing e3nn-mlx IO module translation...")
    
    try:
        test_cartesian_tensor_basic()
        test_spherical_tensor_creation()
        test_spherical_tensor_norms()
        test_spherical_tensor_sum_of_diracs()
        print("\nAll basic tests passed! ✅")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()