#!/usr/bin/env python3
"""
Test script for Gate activation function.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mlx.core as mx
import numpy as np

from e3nn_mlx.nn import Gate
from e3nn_mlx.o3 import Irreps

def test_gate_basic():
    """Test basic Gate functionality."""
    print("Testing basic Gate functionality...")
    
    # Create a simple gate: 16 scalars -> tanh -> gate 16 vectors
    gate = Gate("16x0o", [mx.tanh], "16x0o", [mx.tanh], "16x1e")
    
    print(f"Gate input irreps: {gate.irreps_in}")
    print(f"Gate output irreps: {gate.irreps_out}")
    
    # Create test input
    batch_size = 4
    input_dim = gate.irreps_in.dim
    x = mx.random.normal((batch_size, input_dim))
    
    print(f"Input shape: {x.shape}")
    
    # Apply gate
    try:
        output = gate(x)
        print(f"Output shape: {output.shape}")
        print(f"Expected output shape: {(batch_size, gate.irreps_out.dim)}")
        
        # Check if output shape is correct
        assert output.shape == (batch_size, gate.irreps_out.dim), f"Shape mismatch: {output.shape} != {(batch_size, gate.irreps_out.dim)}"
        print("✓ Gate basic functionality test passed!")
        return True
    except Exception as e:
        print(f"✗ Gate basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gate_no_gates():
    """Test Gate with no gates (only scalars)."""
    print("\nTesting Gate with no gates...")
    
    # Create a gate with only scalars
    gate = Gate("8x0o", [mx.tanh], "", [], "")
    
    print(f"Gate input irreps: {gate.irreps_in}")
    print(f"Gate output irreps: {gate.irreps_out}")
    
    # Create test input
    batch_size = 4
    input_dim = gate.irreps_in.dim
    x = mx.random.normal((batch_size, input_dim))
    
    print(f"Input shape: {x.shape}")
    
    try:
        output = gate(x)
        print(f"Output shape: {output.shape}")
        print(f"Expected output shape: {(batch_size, gate.irreps_out.dim)}")
        
        # Check if output shape is correct
        assert output.shape == (batch_size, gate.irreps_out.dim), f"Shape mismatch: {output.shape} != {(batch_size, gate.irreps_out.dim)}"
        print("✓ Gate no gates test passed!")
        return True
    except Exception as e:
        print(f"✗ Gate no gates test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gate_mixed():
    """Test Gate with mixed scalar and gated outputs."""
    print("\nTesting Gate with mixed outputs...")
    
    # Create a gate with both scalars and gated outputs
    gate = Gate("8x0o", [mx.tanh], "16x0o", [mx.sigmoid], "8x1e+8x1o")
    
    print(f"Gate input irreps: {gate.irreps_in}")
    print(f"Gate output irreps: {gate.irreps_out}")
    
    # Create test input
    batch_size = 4
    input_dim = gate.irreps_in.dim
    x = mx.random.normal((batch_size, input_dim))
    
    print(f"Input shape: {x.shape}")
    
    try:
        output = gate(x)
        print(f"Output shape: {output.shape}")
        print(f"Expected output shape: {(batch_size, gate.irreps_out.dim)}")
        
        # Check if output shape is correct
        assert output.shape == (batch_size, gate.irreps_out.dim), f"Shape mismatch: {output.shape} != {(batch_size, gate.irreps_out.dim)}"
        print("✓ Gate mixed outputs test passed!")
        return True
    except Exception as e:
        print(f"✗ Gate mixed outputs test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gate_errors():
    """Test Gate error handling."""
    print("\nTesting Gate error handling...")
    
    try:
        # Test invalid irreps_gates (non-scalar)
        try:
            gate = Gate("8x0o", [mx.tanh], "8x1e", [mx.tanh], "8x1e")
            print("✗ Should have failed with non-scalar gates")
            return False
        except ValueError as e:
            print(f"✓ Correctly caught non-scalar gates error: {e}")
        
        # Test invalid irreps_scalars (non-scalar)
        try:
            gate = Gate("8x1e", [mx.tanh], "8x0o", [mx.tanh], "8x1e")
            print("✗ Should have failed with non-scalar scalars")
            return False
        except ValueError as e:
            print(f"✓ Correctly caught non-scalar scalars error: {e}")
        
        # Test mismatched irreps_gates and irreps_gated
        try:
            gate = Gate("8x0o", [mx.tanh], "16x0o", [mx.tanh], "8x1e")
            print("✗ Should have failed with mismatched gates and gated")
            return False
        except ValueError as e:
            print(f"✓ Correctly caught mismatched gates and gated error: {e}")
        
        print("✓ Gate error handling test passed!")
        return True
    except Exception as e:
        print(f"✗ Gate error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=== Testing Gate Activation Function ===")
    
    tests = [
        test_gate_basic,
        test_gate_no_gates,
        test_gate_mixed,
        test_gate_errors
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test Results: {passed}/{total} passed ===")
    
    if passed == total:
        print("🎉 All Gate tests passed!")
        return True
    else:
        print("❌ Some Gate tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)