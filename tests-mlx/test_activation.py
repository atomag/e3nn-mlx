#!/usr/bin/env python3
"""
Test script for Activation functions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mlx.core as mx
import numpy as np

from e3nn_mlx.nn import Activation
from e3nn_mlx.o3 import Irreps

def test_activation_basic():
    """Test basic Activation functionality."""
    print("Testing basic Activation functionality...")
    
    # Create a simple activation for scalars
    activation = Activation("16x0o", [mx.tanh])
    
    print(f"Activation input irreps: {activation.irreps_in}")
    print(f"Activation output irreps: {activation.irreps_out}")
    
    # Create test input
    batch_size = 4
    input_dim = activation.irreps_in.dim
    x = mx.random.normal((batch_size, input_dim))
    
    print(f"Input shape: {x.shape}")
    
    try:
        output = activation(x)
        print(f"Output shape: {output.shape}")
        print(f"Expected output shape: {(batch_size, activation.irreps_out.dim)}")
        
        # Check if output shape is correct
        assert output.shape == (batch_size, activation.irreps_out.dim), f"Shape mismatch: {output.shape} != {(batch_size, activation.irreps_out.dim)}"
        print("✓ Activation basic functionality test passed!")
    except Exception as e:
        print(f"✗ Activation basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_activation_mixed():
    """Test Activation with mixed scalar and non-scalar inputs."""
    print("\nTesting Activation with mixed inputs...")
    
    # Create activation with mixed scalars and non-scalars
    activation = Activation("8x0o + 4x1e", [mx.tanh, None])
    
    print(f"Activation input irreps: {activation.irreps_in}")
    print(f"Activation output irreps: {activation.irreps_out}")
    
    # Create test input
    batch_size = 4
    input_dim = activation.irreps_in.dim
    x = mx.random.normal((batch_size, input_dim))
    
    print(f"Input shape: {x.shape}")
    
    try:
        output = activation(x)
        print(f"Output shape: {output.shape}")
        
        # Check if output shape is correct
        assert output.shape == (batch_size, activation.irreps_out.dim), f"Shape mismatch: {output.shape} != {(batch_size, activation.irreps_out.dim)}"
        print("✓ Activation mixed inputs test passed!")
    except Exception as e:
        print(f"✗ Activation mixed inputs test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_activation_multiple_activations():
    """Test Activation with multiple different activation functions."""
    print("\nTesting Activation with multiple activation functions...")
    
    # Create activation with multiple different functions
    activation = Activation("4x0o + 4x0e + 4x1o", [mx.tanh, mx.sigmoid, None])
    
    print(f"Activation input irreps: {activation.irreps_in}")
    print(f"Activation output irreps: {activation.irreps_out}")
    
    # Create test input
    batch_size = 4
    input_dim = activation.irreps_in.dim
    x = mx.random.normal((batch_size, input_dim))
    
    print(f"Input shape: {x.shape}")
    
    try:
        output = activation(x)
        print(f"Output shape: {output.shape}")
        
        # Check if output shape is correct
        assert output.shape == (batch_size, activation.irreps_out.dim), f"Shape mismatch: {output.shape} != {(batch_size, activation.irreps_out.dim)}"
        print("✓ Activation multiple functions test passed!")
    except Exception as e:
        print(f"✗ Activation multiple functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_activation_parity_detection():
    """Test Activation parity detection."""
    print("\nTesting Activation parity detection...")
    
    # Test even function (sigmoid)
    activation_even = Activation("4x0o", [mx.sigmoid])
    print(f"Even activation input: {activation_even.irreps_in}")
    print(f"Even activation output: {activation_even.irreps_out}")
    
    # Test odd function (tanh)
    activation_odd = Activation("4x0o", [mx.tanh])
    print(f"Odd activation input: {activation_odd.irreps_in}")
    print(f"Odd activation output: {activation_odd.irreps_out}")
    
    # Create test input
    batch_size = 4
    input_dim = activation_even.irreps_in.dim
    x = mx.random.normal((batch_size, input_dim))
    
    try:
        output_even = activation_even(x)
        output_odd = activation_odd(x)
        
        print(f"Even activation output shape: {output_even.shape}")
        print(f"Odd activation output shape: {output_odd.shape}")
        
        # Check if output shapes are correct
        assert output_even.shape == (batch_size, activation_even.irreps_out.dim), f"Shape mismatch: {output_even.shape} != {(batch_size, activation_even.irreps_out.dim)}"
        assert output_odd.shape == (batch_size, activation_odd.irreps_out.dim), f"Shape mismatch: {output_odd.shape} != {(batch_size, activation_odd.irreps_out.dim)}"
        print("✓ Activation parity detection test passed!")
    except Exception as e:
        print(f"✗ Activation parity detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_activation_no_activations():
    """Test Activation with no activation functions."""
    print("\nTesting Activation with no activation functions...")
    
    # Create activation with no activations
    activation = Activation("4x0o + 4x1e", [None, None])
    
    print(f"Activation input irreps: {activation.irreps_in}")
    print(f"Activation output irreps: {activation.irreps_out}")
    
    # Create test input
    batch_size = 4
    input_dim = activation.irreps_in.dim
    x = mx.random.normal((batch_size, input_dim))
    
    print(f"Input shape: {x.shape}")
    
    try:
        output = activation(x)
        print(f"Output shape: {output.shape}")
        
        # Check if output shape is correct
        assert output.shape == (batch_size, activation.irreps_out.dim), f"Shape mismatch: {output.shape} != {(batch_size, activation.irreps_out.dim)}"
        print("✓ Activation no activations test passed!")
    except Exception as e:
        print(f"✗ Activation no activations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_activation_errors():
    """Test Activation error handling."""
    print("\nTesting Activation error handling...")
    
    try:
        # Test mismatched irreps and activations
        try:
            activation = Activation("4x0o + 4x1e", [mx.tanh])
            print("✗ Should have failed with mismatched irreps and activations")
            return False
        except ValueError as e:
            print(f"✓ Correctly caught mismatched irreps and activations error: {e}")
        
        # Test activation on non-scalar
        try:
            activation = Activation("4x1e", [mx.tanh])
            print("✗ Should have failed with activation on non-scalar")
            return False
        except ValueError as e:
            print(f"✓ Correctly caught activation on non-scalar error: {e}")
        
        # Test parity violation (neither even nor odd function)
        try:
            def custom_activation(x):
                return x + 0.1 * x * x  # Neither even nor odd
            
            activation = Activation("4x0o", [custom_activation])
            print("✗ Should have failed with parity violation")
            return False
        except ValueError as e:
            print(f"✓ Correctly caught parity violation error: {e}")
        
        print("✓ Activation error handling test passed!")
        return True
    except Exception as e:
        print(f"✗ Activation error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_activation_different_dimensions():
    """Test Activation with different input dimensions."""
    print("\nTesting Activation with different input dimensions...")
    
    # Test with different batch sizes and dimensions
    shapes = [(2, 16), (4, 32), (8, 64)]
    
    activation = Activation("8x0o + 8x1e", [mx.tanh, None])
    
    for shape in shapes:
        print(f"Testing shape: {shape}")
        try:
            x = mx.random.normal(shape)
            output = activation(x)
            print(f"  Output shape: {output.shape}")
            
            # Check if output shape is correct
            expected_shape = (shape[0], activation.irreps_out.dim)
            assert output.shape == expected_shape, f"Shape mismatch: {output.shape} != {expected_shape}"
            print(f"  ✓ Shape {shape} test passed!")
        except Exception as e:
            print(f"  ✗ Shape {shape} test failed: {e}")
            return False
    
    print("✓ Activation different dimensions test passed!")
    return True

def main():
    """Run all tests."""
    print("=== Testing Activation Functions ===")
    
    tests = [
        test_activation_basic,
        test_activation_mixed,
        test_activation_multiple_activations,
        test_activation_parity_detection,
        test_activation_no_activations,
        test_activation_errors,
        test_activation_different_dimensions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test Results: {passed}/{total} passed ===")
    
    if passed == total:
        print("🎉 All Activation tests passed!")
        return True
    else:
        print("❌ Some Activation tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)