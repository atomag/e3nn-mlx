#!/usr/bin/env python3
"""
Test script for BatchNorm.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mlx.core as mx
import numpy as np

from e3nn_mlx.nn import BatchNorm
from e3nn_mlx.o3 import Irreps

def test_batchnorm_basic():
    """Test basic BatchNorm functionality."""
    print("Testing basic BatchNorm functionality...")
    
    # Create a simple batch norm for scalars and vectors
    irreps = Irreps("4x0e + 4x1o")
    bn = BatchNorm(irreps, affine=True)
    
    print(f"BatchNorm irreps: {bn.irreps}")
    print(f"Running mean shape: {bn.running_mean.shape if bn.running_mean is not None else None}")
    print(f"Running var shape: {bn.running_var.shape if bn.running_var is not None else None}")
    print(f"Weight shape: {bn.weight.shape if bn.weight is not None else None}")
    print(f"Bias shape: {bn.bias.shape if bn.bias is not None else None}")
    
    # Create test input
    batch_size = 8
    input_dim = bn.irreps.dim
    x = mx.random.normal((batch_size, input_dim))
    
    print(f"Input shape: {x.shape}")
    
    # Test training mode
    bn.train()
    try:
        output = bn(x)
        print(f"Training output shape: {output.shape}")
        print(f"Expected output shape: {(batch_size, input_dim)}")
        
        # Check if output shape is correct
        assert output.shape == (batch_size, input_dim), f"Shape mismatch: {output.shape} != {(batch_size, input_dim)}"
        print("✓ BatchNorm training test passed!")
    except Exception as e:
        print(f"✗ BatchNorm training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test eval mode
    bn.eval()
    try:
        output = bn(x)
        print(f"Eval output shape: {output.shape}")
        
        # Check if output shape is correct
        assert output.shape == (batch_size, input_dim), f"Shape mismatch: {output.shape} != {(batch_size, input_dim)}"
        print("✓ BatchNorm eval test passed!")
    except Exception as e:
        print(f"✗ BatchNorm eval test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_batchnorm_instance():
    """Test BatchNorm with instance normalization."""
    print("\nTesting BatchNorm with instance normalization...")
    
    # Create instance norm
    irreps = Irreps("4x0e + 4x1o")
    bn = BatchNorm(irreps, affine=True, instance=True)
    
    print(f"Instance BatchNorm irreps: {bn.irreps}")
    print(f"Running mean: {bn.running_mean}")
    print(f"Running var: {bn.running_var}")
    
    # Create test input
    batch_size = 8
    input_dim = bn.irreps.dim
    x = mx.random.normal((batch_size, input_dim))
    
    print(f"Input shape: {x.shape}")
    
    try:
        output = bn(x)
        print(f"Output shape: {output.shape}")
        
        # Check if output shape is correct
        assert output.shape == (batch_size, input_dim), f"Shape mismatch: {output.shape} != {(batch_size, input_dim)}"
        print("✓ BatchNorm instance test passed!")
    except Exception as e:
        print(f"✗ BatchNorm instance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_batchnorm_no_affine():
    """Test BatchNorm without affine parameters."""
    print("\nTesting BatchNorm without affine parameters...")
    
    # Create batch norm without affine
    irreps = Irreps("4x0e + 4x1o")
    bn = BatchNorm(irreps, affine=False)
    
    print(f"No-affine BatchNorm irreps: {bn.irreps}")
    print(f"Weight: {bn.weight}")
    print(f"Bias: {bn.bias}")
    
    # Create test input
    batch_size = 8
    input_dim = bn.irreps.dim
    x = mx.random.normal((batch_size, input_dim))
    
    print(f"Input shape: {x.shape}")
    
    try:
        output = bn(x)
        print(f"Output shape: {output.shape}")
        
        # Check if output shape is correct
        assert output.shape == (batch_size, input_dim), f"Shape mismatch: {output.shape} != {(batch_size, input_dim)}"
        print("✓ BatchNorm no-affine test passed!")
    except Exception as e:
        print(f"✗ BatchNorm no-affine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_batchnorm_different_reductions():
    """Test BatchNorm with different reduction methods."""
    print("\nTesting BatchNorm with different reduction methods...")
    
    # Test mean reduction
    irreps = Irreps("4x0e + 4x1o")
    bn_mean = BatchNorm(irreps, reduce="mean")
    
    # Test max reduction
    bn_max = BatchNorm(irreps, reduce="max")
    
    # Create test input
    batch_size = 8
    input_dim = irreps.dim
    x = mx.random.normal((batch_size, input_dim))
    
    print(f"Input shape: {x.shape}")
    
    try:
        output_mean = bn_mean(x)
        output_max = bn_max(x)
        
        print(f"Mean reduction output shape: {output_mean.shape}")
        print(f"Max reduction output shape: {output_max.shape}")
        
        # Check if output shapes are correct
        assert output_mean.shape == (batch_size, input_dim), f"Shape mismatch: {output_mean.shape} != {(batch_size, input_dim)}"
        assert output_max.shape == (batch_size, input_dim), f"Shape mismatch: {output_max.shape} != {(batch_size, input_dim)}"
        print("✓ BatchNorm different reductions test passed!")
    except Exception as e:
        print(f"✗ BatchNorm different reductions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_batchnorm_normalization_methods():
    """Test BatchNorm with different normalization methods."""
    print("\nTesting BatchNorm with different normalization methods...")
    
    # Test norm normalization
    irreps = Irreps("4x0e + 4x1o")
    bn_norm = BatchNorm(irreps, normalization="norm")
    
    # Test component normalization
    bn_component = BatchNorm(irreps, normalization="component")
    
    # Create test input
    batch_size = 8
    input_dim = irreps.dim
    x = mx.random.normal((batch_size, input_dim))
    
    print(f"Input shape: {x.shape}")
    
    try:
        output_norm = bn_norm(x)
        output_component = bn_component(x)
        
        print(f"Norm normalization output shape: {output_norm.shape}")
        print(f"Component normalization output shape: {output_component.shape}")
        
        # Check if output shapes are correct
        assert output_norm.shape == (batch_size, input_dim), f"Shape mismatch: {output_norm.shape} != {(batch_size, input_dim)}"
        assert output_component.shape == (batch_size, input_dim), f"Shape mismatch: {output_component.shape} != {(batch_size, input_dim)}"
        print("✓ BatchNorm different normalization methods test passed!")
    except Exception as e:
        print(f"✗ BatchNorm different normalization methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_batchnorm_complex_irreps():
    """Test BatchNorm with complex irreps."""
    print("\nTesting BatchNorm with complex irreps...")
    
    # Create batch norm for complex irreps
    irreps = Irreps("2x0e + 3x1o + 1x2e + 2x1e")
    bn = BatchNorm(irreps, affine=True)
    
    print(f"Complex BatchNorm irreps: {bn.irreps}")
    print(f"Running mean shape: {bn.running_mean.shape if bn.running_mean is not None else None}")
    print(f"Running var shape: {bn.running_var.shape if bn.running_var is not None else None}")
    print(f"Weight shape: {bn.weight.shape if bn.weight is not None else None}")
    print(f"Bias shape: {bn.bias.shape if bn.bias is not None else None}")
    
    # Create test input
    batch_size = 8
    input_dim = bn.irreps.dim
    x = mx.random.normal((batch_size, input_dim))
    
    print(f"Input shape: {x.shape}")
    
    try:
        output = bn(x)
        print(f"Output shape: {output.shape}")
        
        # Check if output shape is correct
        assert output.shape == (batch_size, input_dim), f"Shape mismatch: {output.shape} != {(batch_size, input_dim)}"
        print("✓ BatchNorm complex irreps test passed!")
    except Exception as e:
        print(f"✗ BatchNorm complex irreps test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_batchnorm_errors():
    """Test BatchNorm error handling."""
    print("\nTesting BatchNorm error handling...")
    
    try:
        # Test invalid reduce parameter
        try:
            bn = BatchNorm("4x0e", reduce="invalid")
            print("✗ Should have failed with invalid reduce parameter")
            return False
        except ValueError as e:
            print(f"✓ Correctly caught invalid reduce parameter error: {e}")
        
        # Test invalid normalization parameter
        try:
            bn = BatchNorm("4x0e", normalization="invalid")
            print("✗ Should have failed with invalid normalization parameter")
            return False
        except ValueError as e:
            print(f"✓ Correctly caught invalid normalization parameter error: {e}")
        
        print("✓ BatchNorm error handling test passed!")
        return True
    except Exception as e:
        print(f"✗ BatchNorm error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=== Testing BatchNorm ===")
    
    tests = [
        test_batchnorm_basic,
        test_batchnorm_instance,
        test_batchnorm_no_affine,
        test_batchnorm_different_reductions,
        test_batchnorm_normalization_methods,
        test_batchnorm_complex_irreps,
        test_batchnorm_errors
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test Results: {passed}/{total} passed ===")
    
    if passed == total:
        print("🎉 All BatchNorm tests passed!")
        return True
    else:
        print("❌ Some BatchNorm tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)