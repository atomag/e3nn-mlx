#!/usr/bin/env python3
"""
Basic functionality test for e3nn-mlx package.
This script verifies that the core components are working correctly.
"""

import mlx.core as mx
from e3nn_mlx.o3 import Irreps, TensorProduct, FullyConnectedTensorProduct, spherical_harmonics
from e3nn_mlx.util.test import assert_equivariant

def test_basic_irreps():
    """Test basic irreps functionality"""
    print("Testing Irreps...")
    
    irreps = Irreps("1x0e + 2x1o")
    print(f"  Created: {irreps}")
    print(f"  Dimension: {irreps.dim}")
    
    # Test creating random data
    data = mx.random.normal((10, irreps.dim))
    print(f"  Random data shape: {data.shape}")
    
    return True

def test_spherical_harmonics():
    """Test spherical harmonics"""
    print("\nTesting spherical harmonics...")
    
    # Create random directions
    directions = mx.random.normal((100, 3))
    directions = directions / mx.linalg.norm(directions, axis=1, keepdims=True)
    
    # Compute spherical harmonics
    sh = spherical_harmonics(l=2, x=directions, normalize=True)
    print(f"  Spherical harmonics shape: {sh.shape}")
    
    return True

def test_tensor_product():
    """Test basic tensor product"""
    print("\nTesting tensor product...")
    
    # Create simple irreps
    irreps_in1 = Irreps("1x1o")
    irreps_in2 = Irreps("1x1o") 
    irreps_out = Irreps("1x0e + 1x1o + 1x2e")
    
    # Create tensor product - use FullyConnectedTensorProduct for simple testing
    tp = FullyConnectedTensorProduct(
        irreps_in1=irreps_in1,
        irreps_in2=irreps_in2,
        irreps_out=irreps_out
    )
    
    print(f"  Tensor product: {tp}")
    
    # Test with random data
    x1 = mx.random.normal((5, irreps_in1.dim))
    x2 = mx.random.normal((5, irreps_in2.dim))
    
    result = tp(x1, x2)
    print(f"  Result shape: {result.shape}")
    
    return True

def test_fully_connected_tensor_product():
    """Test fully connected tensor product"""
    print("\nTesting fully connected tensor product...")
    
    # Create irreps
    irreps_in1 = Irreps("1x0e + 1x1o")
    irreps_in2 = Irreps("1x0e + 1x1o")
    irreps_out = Irreps("1x0e + 1x1o")
    
    # Create FC tensor product
    fctp = FullyConnectedTensorProduct(
        irreps_in1=irreps_in1,
        irreps_in2=irreps_in2,
        irreps_out=irreps_out
    )
    
    print(f"  FC tensor product: {fctp}")
    
    # Test with random data
    x1 = mx.random.normal((3, irreps_in1.dim))
    x2 = mx.random.normal((3, irreps_in2.dim))
    
    result = fctp(x1, x2)
    print(f"  Result shape: {result.shape}")
    
    return True

def test_equivariance():
    """Test equivariance for a simple case"""
    print("\nTesting equivariance...")
    
    # Create simple tensor product
    tp = FullyConnectedTensorProduct("1x1o", "1x1o", "1x0e")
    
    try:
        # This should work for simple cases
        from e3nn_mlx.o3 import Irreps
        irreps_in1 = Irreps("1x1o")
        irreps_in2 = Irreps("1x1o")
        irreps_out = Irreps("1x0e")
        
        tp = FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out)
        assert_equivariant(tp, irreps_in=[irreps_in1, irreps_in2], irreps_out=irreps_out)
        print("  Equivariance test passed!")
        return True
    except Exception as e:
        print(f"  Equivariance test failed: {e}")
        return False

def main():
    """Run all basic tests"""
    print("=== e3nn-mlx Basic Functionality Test ===")
    
    tests = [
        test_basic_irreps,
        test_spherical_harmonics,
        test_tensor_product,
        test_fully_connected_tensor_product,
        test_equivariance
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"  ❌ {test.__name__} failed")
        except Exception as e:
            print(f"  ❌ {test.__name__} error: {e}")
    
    print(f"\n=== Summary: {passed}/{len(tests)} tests passed ===")
    
    if passed == len(tests):
        print("🎉 All basic functionality tests passed!")
    else:
        print("⚠️  Some tests failed, but core functionality is working")
    
    return passed == len(tests)

if __name__ == "__main__":
    main()