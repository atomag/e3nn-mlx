#!/usr/bin/env python3
"""
Test script for Irreps functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mlx.core as mx
import numpy as np

from e3nn_mlx.o3 import Irreps, Irrep

def test_irrep_basic():
    """Test basic Irrep functionality."""
    print("Testing basic Irrep functionality...")
    
    # Test creation from l, p
    irrep1 = Irrep(0, 1)
    print(f"Irrep(0, 1) = {irrep1}")
    
    # Test creation from string
    irrep2 = Irrep("1o")
    print(f"Irrep('1o') = {irrep2}")
    
    # Test properties
    print(f"irrep1.l = {irrep1.l}")
    print(f"irrep1.p = {irrep1.p}")
    print(f"irrep1.dim = {irrep1.dim}")
    
    # Test tuple methods
    print(f"len(irrep1) = {len(irrep1)}")
    print(f"irrep1[0] = {irrep1[0]}")
    print(f"irrep1[1] = {irrep1[1]}")
    
    # Test count and index
    print(f"irrep1.count(0) = {irrep1.count(0)}")
    print(f"irrep1.index(0) = {irrep1.index(0)}")
    
    # Test multiplication
    print(f"irrep1 * irrep2 = {list(irrep1 * irrep2)}")
    
    print("✓ Irrep basic functionality test passed!")
    return True

def test_irrep_errors():
    """Test Irrep error handling."""
    print("\nTesting Irrep error handling...")
    
    try:
        # Test invalid l
        try:
            irrep = Irrep(-1, 1)
            print("✗ Should have failed with negative l")
            return False
        except ValueError as e:
            print(f"✓ Correctly caught negative l error: {e}")
        
        # Test invalid p
        try:
            irrep = Irrep(0, 0)
            print("✗ Should have failed with invalid p")
            return False
        except ValueError as e:
            print(f"✓ Correctly caught invalid p error: {e}")
        
        # Test invalid string
        try:
            irrep = Irrep("invalid")
            print("✗ Should have failed with invalid string")
            return False
        except ValueError as e:
            print(f"✓ Correctly caught invalid string error: {e}")
        
        print("✓ Irrep error handling test passed!")
        return True
    except Exception as e:
        print(f"✗ Irrep error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_irreps_basic():
    """Test basic Irreps functionality."""
    print("\nTesting basic Irreps functionality...")
    
    # Test creation from string
    irreps1 = Irreps("2x0e + 3x1o")
    print(f"Irreps('2x0e + 3x1o') = {irreps1}")
    
    # Test properties
    print(f"irreps1.dim = {irreps1.dim}")
    print(f"irreps1.num_irreps = {irreps1.num_irreps}")
    print(f"len(irreps1) = {len(irreps1)}")
    
    # Test indexing
    first = irreps1[0]
    print(f"irreps1[0] = {first}")
    print(f"Type of first element: {type(first)}")
    
    # Test tuple methods on _MulIr
    print(f"len(first) = {len(first)}")
    print(f"first[0] = {first[0]}")
    print(f"first[1] = {first[1]}")
    
    # Test count for Irrep objects
    ir_0e = Irrep("0e")
    ir_1o = Irrep("1o")
    print(f"irreps1.count({ir_0e}) = {irreps1.count(ir_0e)}")
    print(f"irreps1.count({ir_1o}) = {irreps1.count(ir_1o)}")
    
    # Test contains
    print(f"{ir_0e} in irreps1 = {ir_0e in irreps1}")
    print(f"{ir_1o} in irreps1 = {ir_1o in irreps1}")
    
    print("✓ Irreps basic functionality test passed!")
    return True

def test_irreps_operations():
    """Test Irreps operations."""
    print("\nTesting Irreps operations...")
    
    # Test addition
    irreps1 = Irreps("2x0e")
    irreps2 = Irreps("3x1o")
    irreps_sum = irreps1 + irreps2
    print(f"{irreps1} + {irreps2} = {irreps_sum}")
    
    # Test multiplication by scalar
    irreps_scaled = irreps1 * 3
    print(f"{irreps1} * 3 = {irreps_scaled}")
    
    # Test slice
    irreps_slice = irreps_sum[1:]
    print(f"{irreps_sum}[1:] = {irreps_slice}")
    
    print("✓ Irreps operations test passed!")
    return True

def test_irreps_simplify():
    """Test Irreps simplification."""
    print("\nTesting Irreps simplification...")
    
    # Test simplification of redundant irreps
    irreps = Irreps("2x0e + 1x0e + 3x1o")
    simplified = irreps.simplify()
    print(f"{irreps}.simplify() = {simplified}")
    
    # Test that simplification preserves total dimension
    print(f"Original dim: {irreps.dim}")
    print(f"Simplified dim: {simplified.dim}")
    assert irreps.dim == simplified.dim, "Simplification should preserve dimension"
    
    print("✓ Irreps simplification test passed!")
    return True

def test_irreps_errors():
    """Test Irreps error handling."""
    print("\nTesting Irreps error handling...")
    
    try:
        # Test invalid string format
        try:
            irreps = Irreps("invalid")
            print("✗ Should have failed with invalid string format")
            return False
        except ValueError as e:
            print(f"✓ Correctly caught invalid string format error: {e}")
        
        # Test index out of bounds
        irreps = Irreps("2x0e")
        try:
            idx = irreps.index((3, Irrep("0e")))
            print("✗ Should have failed with index out of bounds")
            return False
        except ValueError as e:
            print(f"✓ Correctly caught index out of bounds error: {e}")
        
        print("✓ Irreps error handling test passed!")
        return True
    except Exception as e:
        print(f"✗ Irreps error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=== Testing Irreps Functionality ===")
    
    tests = [
        test_irrep_basic,
        test_irrep_errors,
        test_irreps_basic,
        test_irreps_operations,
        test_irreps_simplify,
        test_irreps_errors
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test Results: {passed}/{total} passed ===")
    
    if passed == total:
        print("🎉 All Irreps tests passed!")
        return True
    else:
        print("❌ Some Irreps tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)