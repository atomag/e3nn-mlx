#!/usr/bin/env python3
"""
Test script for validation utilities.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mlx.core as mx
import numpy as np

from e3nn_mlx.util._validation import (
    validate_type, validate_range, validate_array_shape,
    validate_array_dimensions, validate_string_choice, validate_boolean,
    ValidationError
)
import builtins

def test_validate_type():
    """Test type validation."""
    print("Testing type validation...")
    
    # Test valid types
    try:
        validate_type(5, int, "test_param")
        validate_type("hello", str, "test_param")
        validate_type([1, 2, 3], list, "test_param")
        validate_type(mx.array([1, 2, 3]), mx.array, "test_param")
        print("✓ Valid type validation passed")
    except Exception as e:
        print(f"✗ Valid type validation failed: {e}")
        return False
    
    # Test invalid types
    try:
        validate_type(5, str, "test_param")
        print("✗ Should have failed with invalid type")
        return False
    except builtins.TypeError as e:
        print(f"✓ Correctly caught invalid type error: {e}")
    
    return True

def test_validate_range():
    """Test range validation."""
    print("\nTesting range validation...")
    
    # Test valid ranges
    try:
        validate_range(5, "test_param", 0, 10)
        validate_range(0, "test_param", 0, 10, inclusive_min=True)
        validate_range(10, "test_param", 0, 10, inclusive_max=True)
        print("✓ Valid range validation passed")
    except Exception as e:
        print(f"✗ Valid range validation failed: {e}")
        return False
    
    # Test invalid ranges
    try:
        validate_range(15, "test_param", 0, 10)
        print("✗ Should have failed with value too high")
        return False
    except builtins.ValueError as e:
        print(f"✓ Correctly caught value too high error: {e}")
    
    try:
        validate_range(-1, "test_param", 0, 10)
        print("✗ Should have failed with value too low")
        return False
    except builtins.ValueError as e:
        print(f"✓ Correctly caught value too low error: {e}")
    
    return True

def test_validate_array_shape():
    """Test array shape validation."""
    print("\nTesting array shape validation...")
    
    # Test valid shapes
    try:
        arr = mx.array([[1, 2, 3], [4, 5, 6]])
        validate_array_shape(arr, "test_param", (2, 3))
        validate_array_shape(arr, "test_param", (2, None))
        print("✓ Valid array shape validation passed")
    except Exception as e:
        print(f"✗ Valid array shape validation failed: {e}")
        return False
    
    # Test invalid shapes
    try:
        arr = mx.array([[1, 2, 3], [4, 5, 6]])
        validate_array_shape(arr, "test_param", (3, 3))
        print("✗ Should have failed with wrong shape")
        return False
    except builtins.ValueError as e:
        print(f"✓ Correctly caught wrong shape error: {e}")
    
    return True

def test_validate_array_dimensions():
    """Test array dimensions validation."""
    print("\nTesting array dimensions validation...")
    
    # Test valid dimensions
    try:
        arr = mx.array([[1, 2, 3], [4, 5, 6]])
        validate_array_dimensions(arr, "test_param", 2)
        print("✓ Valid array dimensions validation passed")
    except Exception as e:
        print(f"✗ Valid array dimensions validation failed: {e}")
        return False
    
    # Test invalid dimensions
    try:
        arr = mx.array([[1, 2, 3], [4, 5, 6]])
        validate_array_dimensions(arr, "test_param", 1)
        print("✗ Should have failed with wrong dimensions")
        return False
    except builtins.ValueError as e:
        print(f"✓ Correctly caught wrong dimensions error: {e}")
    
    return True

def test_validate_string_choice():
    """Test string choice validation."""
    print("\nTesting string choice validation...")
    
    # Test valid choices
    try:
        validate_string_choice("mean", "test_param", ["mean", "max"])
        validate_string_choice("sigmoid", "test_param", ["tanh", "sigmoid", "relu"])
        print("✓ Valid string choice validation passed")
    except Exception as e:
        print(f"✗ Valid string choice validation failed: {e}")
        return False
    
    # Test invalid choices
    try:
        validate_string_choice("invalid", "test_param", ["mean", "max"])
        print("✗ Should have failed with invalid choice")
        return False
    except builtins.ValueError as e:
        print(f"✓ Correctly caught invalid choice error: {e}")
    
    return True

def test_validate_boolean():
    """Test boolean validation."""
    print("\nTesting boolean validation...")
    
    # Test valid booleans
    try:
        validate_boolean(True, "test_param")
        validate_boolean(False, "test_param")
        print("✓ Valid boolean validation passed")
    except Exception as e:
        print(f"✗ Valid boolean validation failed: {e}")
        return False
    
    # Test invalid booleans
    try:
        validate_boolean("true", "test_param")
        print("✗ Should have failed with non-boolean")
        return False
    except builtins.TypeError as e:
        print(f"✓ Correctly caught non-boolean error: {e}")
    
    return True

def test_exception_hierarchy():
    """Test exception hierarchy."""
    print("\nTesting exception hierarchy...")
    
    try:
        raise ValidationError("Test validation error")
    except Exception as e:
        print(f"✓ ValidationError caught as: {type(e).__name__}: {e}")
    
    try:
        raise TypeError("test_param", "Test type error")
    except ValidationError as e:
        print(f"✓ TypeError caught as ValidationError: {type(e).__name__}: {e}")
    
    try:
        raise ValueError("test_param", "Test value error", "additional info")
    except ValidationError as e:
        print(f"✓ ValueError caught as ValidationError: {type(e).__name__}: {e}")
    
    return True

def test_integration_with_batchnorm():
    """Test integration with BatchNorm validation."""
    print("\nTesting integration with BatchNorm...")
    
    from e3nn_mlx.nn import BatchNorm
    from e3nn_mlx.o3 import Irreps
    
    # Test valid parameters
    try:
        bn = BatchNorm("4x0e + 4x1o", eps=1e-5, momentum=0.1, affine=True)
        print("✓ Valid BatchNorm parameters passed")
    except Exception as e:
        print(f"✗ Valid BatchNorm parameters failed: {e}")
        return False
    
    # Test invalid parameters
    try:
        bn = BatchNorm("4x0e + 4x1o", eps=-1.0)
        print("✗ Should have failed with negative eps")
        return False
    except builtins.ValueError as e:
        print(f"✓ Correctly caught negative eps error: {e}")
    
    return True

def main():
    """Run all tests."""
    print("=== Testing Validation Utilities ===")
    
    tests = [
        test_validate_type,
        test_validate_range,
        test_validate_array_shape,
        test_validate_array_dimensions,
        test_validate_string_choice,
        test_validate_boolean,
        test_exception_hierarchy,
        test_integration_with_batchnorm
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test Results: {passed}/{total} passed ===")
    
    if passed == total:
        print("🎉 All validation tests passed!")
        return True
    else:
        print("❌ Some validation tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)