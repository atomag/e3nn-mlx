#!/usr/bin/env python3
"""Test runner for e3nn-mlx tests."""

import os
import sys
import pytest
import traceback


def run_tests():
    """Run all available tests."""
    print("🧪 E3NN-MLX Test Suite")
    print("=" * 50)
    
    # Test files to run
    test_files = [
        "defaults_test.py",
        "o3/irreps_test.py",
        "o3/linear_test.py",
        "o3/rotation_test.py", 
        "o3/tensor_product_test.py",
        "math/perm_test.py",
    ]
    
    results = {}
    
    for test_file in test_files:
        print(f"\n📋 Running {test_file}...")
        try:
            # Change to tests-mlx directory
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            
            # Run pytest on this specific file
            exit_code = pytest.main([test_file, "-v", "-s"])
            
            if exit_code == 0:
                print(f"✅ {test_file} PASSED")
                results[test_file] = "PASSED"
            else:
                print(f"❌ {test_file} FAILED (exit code: {exit_code})")
                results[test_file] = "FAILED"
                
        except Exception as e:
            print(f"❌ {test_file} ERROR: {e}")
            traceback.print_exc()
            results[test_file] = "ERROR"
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for r in results.values() if r == "PASSED")
    total = len(results)
    
    for test_file, result in results.items():
        status = "✅" if result == "PASSED" else "❌"
        print(f"{status} {test_file}: {result}")
    
    print(f"\n📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"⚠️  {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())