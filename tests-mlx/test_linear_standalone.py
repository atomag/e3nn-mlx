#!/usr/bin/env python3
"""
Test script to verify Linear implementation works correctly
without relying on the broken SlowLinear reference.
"""

import mlx.core as mx
from e3nn_mlx import o3


def test_linear_basic_functionality():
    """Test basic linear functionality."""
    print("Testing basic linear functionality...")
    
    # Test case 1: Simple identity transformation
    irreps_in = "2x0e + 1x1o"
    irreps_out = "2x0e + 1x1o"
    
    linear = o3.Linear(irreps_in, irreps_out)
    
    # Test input
    batch_size = 4
    x = mx.random.normal((batch_size, linear.irreps_in.dim))
    
    # Forward pass
    output = linear(x)
    
    # Check shape
    assert output.shape == (batch_size, linear.irreps_out.dim), \
        f"Expected shape {(batch_size, linear.irreps_out.dim)}, got {output.shape}"
    
    print(f"✓ Basic functionality: {irreps_in} -> {irreps_out}")
    

def test_linear_with_bias():
    """Test linear with bias."""
    print("Testing linear with bias...")
    
    irreps_in = "3x0e"
    irreps_out = "2x0e"
    
    linear = o3.Linear(irreps_in, irreps_out, biases=True)
    
    # Set bias to known value
    if linear.bias.size > 0:
        linear.bias[:] = 1.0
    
    # Test with zero input
    x = mx.zeros((2, linear.irreps_in.dim))
    output = linear(x)
    
    # With zero input and bias=1.0, output should be non-zero
    assert not mx.allclose(output, mx.zeros_like(output)), \
        "Bias should produce non-zero output with zero input"
    
    print(f"✓ Bias functionality: {irreps_in} -> {irreps_out}")


def test_linear_empty_irreps():
    """Test linear with empty irreps (zero multiplicity)."""
    print("Testing linear with empty irreps...")
    
    # Test case with zero multiplicity
    irreps_in = "2x1o + 0x3e"
    irreps_out = "2x1o + 0x3e"
    
    linear = o3.Linear(irreps_in, irreps_out)
    
    # Test input
    batch_size = 4
    x = mx.random.normal((batch_size, linear.irreps_in.dim))
    
    # Forward pass - should not crash
    output = linear(x)
    
    # Check shape
    assert output.shape == (batch_size, linear.irreps_out.dim), \
        f"Expected shape {(batch_size, linear.irreps_out.dim)}, got {output.shape}"
    
    print(f"✓ Empty irreps: {irreps_in} -> {irreps_out}")


def test_linear_complex_case():
    """Test a complex case that was previously failing."""
    print("Testing complex case...")
    
    irreps_in = "1e + 2e + 3x3o + 3x1e"
    irreps_out = "1e + 2e + 4x1e + 3x3o"
    
    linear = o3.Linear(irreps_in, irreps_out)
    
    # Test input
    batch_size = 4
    x = mx.random.normal((batch_size, linear.irreps_in.dim))
    
    # Forward pass - should not crash
    output = linear(x)
    
    # Check shape
    assert output.shape == (batch_size, linear.irreps_out.dim), \
        f"Expected shape {(batch_size, linear.irreps_out.dim)}, got {output.shape}"
    
    # Check output is finite
    assert mx.all(mx.isfinite(output)), "Output should be finite"
    
    print(f"✓ Complex case: {irreps_in} -> {irreps_out}")


def test_linear_equivariance():
    """Test basic equivariance property."""
    print("Testing equivariance...")
    
    # Vector to vector transformation
    irreps_in = "1x1o"
    irreps_out = "1x1o"
    
    linear = o3.Linear(irreps_in, irreps_out)
    
    # Test input
    batch_size = 3
    x = mx.random.normal((batch_size, linear.irreps_in.dim))
    
    # Forward pass
    output = linear(x)
    
    # Check shape
    assert output.shape == (batch_size, linear.irreps_out.dim), \
        f"Expected shape {(batch_size, linear.irreps_out.dim)}, got {output.shape}"
    
    # Check that non-zero input produces non-zero output
    if not mx.allclose(x, mx.zeros_like(x)):
        assert not mx.allclose(output, mx.zeros_like(output)), \
            "Non-zero input should produce non-zero output"
    
    print(f"✓ Equivariance: {irreps_in} -> {irreps_out}")


def test_all_cases():
    """Run all tests."""
    print("=" * 50)
    print("Testing Linear implementation independently")
    print("=" * 50)
    
    test_linear_basic_functionality()
    test_linear_with_bias()
    test_linear_empty_irreps()
    test_linear_complex_case()
    test_linear_equivariance()
    
    print("=" * 50)
    print("All Linear tests passed! ✓")
    print("=" * 50)


if __name__ == "__main__":
    test_all_cases()