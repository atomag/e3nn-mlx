#!/usr/bin/env python3
"""
Simple test for the IO module using only mlx.
"""

import mlx.core as mx
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import the IO module components directly
from e3nn_mlx.io._cartesian_tensor import CartesianTensor


def test_cartesian_tensor():
    """Test Cartesian tensor functionality."""
    print("Testing CartesianTensor...")
    
    try:
        # Test creation
        ct = CartesianTensor("ij=ji")
        print(f"✓ CartesianTensor created: {ct}")
        print(f"✓ Dimension: {ct.dim}")
        
        # Test basic tensor operations
        t = mx.arange(9, dtype=mx.float32).reshape(3, 3)
        print(f"✓ Original tensor:\n{t}")
        
        # Test from_cartesian
        y = ct.from_cartesian(t)
        print(f"✓ To irreps: {y}")
        print(f"✓ Shape: {y.shape}")
        
        # Test to_cartesian
        z = ct.to_cartesian(y)
        print(f"✓ Back to cartesian:\n{z}")
        
        # Test symmetry
        symmetric = (t + t.T) / 2
        close = mx.allclose(z, symmetric, atol=1e-5)
        print(f"✓ Symmetric reconstruction close: {close}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_spherical_tensor_creation():
    """Test SphericalTensor creation."""
    print("\nTesting SphericalTensor creation...")
    
    try:
        from e3nn_mlx.io._spherical_tensor import SphericalTensor
        
        # Test creation
        st = SphericalTensor(1, 1, 1)
        print(f"✓ SphericalTensor(1, 1, 1): {st}")
        print(f"✓ Dimension: {st.dim}")
        
        st_odd = SphericalTensor(1, 1, -1)
        print(f"✓ SphericalTensor(1, 1, -1): {st_odd}")
        print(f"✓ Dimension: {st_odd.dim}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing e3nn-mlx IO module basic functionality...")
    
    success = True
    success &= test_cartesian_tensor()
    success &= test_spherical_tensor_creation()
    
    if success:
        print("\n🎉 All basic tests passed!")
    else:
        print("\n❌ Some tests failed")