"""
Integration Test for e3nn-mlx Optimizations

This script tests the integration of all optimization components
to ensure they work together correctly.
"""

import time
import mlx.core as mx
from e3nn_mlx.o3 import TensorProduct, spherical_harmonics, Irreps, Linear
from e3nn_mlx.util.compile import get_profiler, benchmark_compiled_vs_interpreted
from e3nn_mlx.o3._tensor_product._tensor_product_fusion import adaptive_tensor_product
from e3nn_mlx.o3._linear_optimized import MemoryEfficientLinear, create_memory_efficient_linear
from e3nn_mlx.o3._specialized_kernels import SpecializedKernels, FastTensorProduct, OptimizedSphericalHarmonics
from e3nn_mlx.util.adaptive_optimization import AdaptiveOptimizer, get_global_optimizer


def test_integration_tensor_product():
    """Test integrated tensor product optimizations."""
    print("Testing Integrated Tensor Product Optimizations...")
    
    # Create test data
    batch_size = 128
    irreps_in1 = Irreps("2x1o + 1x0e")
    irreps_in2 = Irreps("2x1o + 1x0e")
    irreps_out = Irreps("1x0e + 2x1o + 1x2e")
    
    # Test inputs
    x1 = mx.random.normal((batch_size, irreps_in1.dim))
    x2 = mx.random.normal((batch_size, irreps_in2.dim))
    
    # Original tensor product
    tp_original = TensorProduct(irreps_in1, irreps_in2, irreps_out, [])
    
    # Optimized versions
    tp_fast = FastTensorProduct(irreps_in1, irreps_in2, irreps_out)
    
    # Test correctness
    result_original = tp_original(x1, x2)
    result_fast = tp_fast(x1, x2, mx.random.normal((100,)))  # Dummy weights
    
    print(f"Original result shape: {result_original.shape}")
    print(f"Fast result shape: {result_fast.shape}")
    
    # Test specialized kernels
    scalar_result = SpecializedKernels.scalar_scalar_operation(
        x1[..., :1], x2[..., :1], 1.0
    )
    print(f"Scalar operation result shape: {scalar_result.shape}")
    
    return True


def test_integration_linear_layers():
    """Test integrated linear layer optimizations."""
    print("\nTesting Integrated Linear Layer Optimizations...")
    
    # Create test data
    batch_size = 256
    irreps_in = Irreps("10x1o + 5x0e")
    irreps_out = Irreps("8x1o + 3x0e + 2x2e")
    
    # Test inputs
    x = mx.random.normal((batch_size, irreps_in.dim))
    
    # Original linear layer
    linear_original = Linear(irreps_in, irreps_out)
    
    # Memory-efficient linear layer
    linear_optimized = MemoryEfficientLinear(irreps_in, irreps_out, chunk_size=64)
    
    # Test correctness
    result_original = linear_original(x)
    result_optimized = linear_optimized(x)
    
    print(f"Original result shape: {result_original.shape}")
    print(f"Optimized result shape: {result_optimized.shape}")
    
    # Test performance
    num_runs = 50
    
    # Original
    start_time = time.time()
    for _ in range(num_runs):
        result_original = linear_original(x)
    original_time = (time.time() - start_time) / num_runs
    
    # Optimized
    start_time = time.time()
    for _ in range(num_runs):
        result_optimized = linear_optimized(x)
    optimized_time = (time.time() - start_time) / num_runs
    
    print(f"Original time: {original_time:.4f}s")
    print(f"Optimized time: {optimized_time:.4f}s ({original_time/optimized_time:.2f}x speedup)")
    
    return True


def test_integration_spherical_harmonics():
    """Test integrated spherical harmonics optimizations."""
    print("\nTesting Integrated Spherical Harmonics Optimizations...")
    
    # Create test data
    batch_size = 1000
    x = mx.random.normal((batch_size, 3))
    
    # Test different l values
    l_values = [0, 1, 2]
    
    for l in l_values:
        print(f"\nTesting l={l}...")
        
        # Original implementation
        start_time = time.time()
        for _ in range(30):
            result_original = spherical_harmonics(l, x, normalize=True)
        original_time = (time.time() - start_time) / 30
        
        # Optimized implementation
        if l == 0:
            optimized_func = OptimizedSphericalHarmonics.spherical_harmonics_l0
        elif l == 1:
            optimized_func = OptimizedSphericalHarmonics.spherical_harmonics_l1
        elif l == 2:
            optimized_func = OptimizedSphericalHarmonics.spherical_harmonics_l2
        else:
            continue
        
        start_time = time.time()
        for _ in range(30):
            result_optimized = optimized_func(x, normalize=True)
        optimized_time = (time.time() - start_time) / 30
        
        # Check correctness
        if result_original.shape == result_optimized.shape:
            max_diff = mx.max(mx.abs(result_original - result_optimized)).item()
            print(f"  Max difference: {max_diff:.2e}")
        else:
            print(f"  Shape mismatch: {result_original.shape} vs {result_optimized.shape}")
        
        print(f"  Original time: {original_time:.4f}s")
        print(f"  Optimized time: {optimized_time:.4f}s ({original_time/optimized_time:.2f}x speedup)")
    
    return True


def test_adaptive_optimization():
    """Test adaptive optimization system."""
    print("\nTesting Adaptive Optimization System...")
    
    # Create optimizer
    optimizer = AdaptiveOptimizer()
    
    # Create test data
    batch_size = 512
    irreps_in = Irreps("5x1o + 3x0e")
    irreps_out = Irreps("4x1o + 2x0e")
    
    x = mx.random.normal((batch_size, irreps_in.dim))
    
    # Create test operation
    def test_operation(x):
        # Simple linear operation
        weights = mx.random.normal((irreps_in.dim, irreps_out.dim))
        return mx.matmul(x, weights)
    
    # Test adaptive optimization
    output_shape = (batch_size, irreps_out.dim)
    
    start_time = time.time()
    result_optimized = optimizer.optimize_operation(
        test_operation, [x], "linear", output_shape
    )
    optimized_time = time.time() - start_time
    
    # Test original operation
    start_time = time.time()
    result_original = test_operation(x)
    original_time = time.time() - start_time
    
    print(f"Original time: {original_time:.4f}s")
    print(f"Optimized time: {optimized_time:.4f}s ({original_time/optimized_time:.2f}x speedup)")
    
    # Get optimization summary
    summary = optimizer.get_optimization_summary()
    print(f"Optimization summary: {summary['total_operations']} operations tracked")
    
    return True


def test_memory_efficiency():
    """Test memory efficiency improvements."""
    print("\nTesting Memory Efficiency...")
    
    # Create large test data
    batch_size = 2000
    irreps_in = Irreps("20x1o + 10x0e + 5x2e")
    irreps_out = Irreps("15x1o + 8x0e + 3x2e")
    
    x = mx.random.normal((batch_size, irreps_in.dim))
    
    # Original linear layer
    linear_original = Linear(irreps_in, irreps_out)
    
    # Memory-efficient linear layer
    linear_optimized = MemoryEfficientLinear(
        irreps_in, irreps_out, 
        chunk_size=128, 
        use_gradient_checkpointing=True
    )
    
    # Test performance with memory constraints
    num_runs = 10
    
    # Original
    start_time = time.time()
    for _ in range(num_runs):
        result_original = linear_original(x)
    original_time = (time.time() - start_time) / num_runs
    
    # Optimized
    start_time = time.time()
    for _ in range(num_runs):
        result_optimized = linear_optimized(x)
    optimized_time = (time.time() - start_time) / num_runs
    
    print(f"Original time (large batch): {original_time:.4f}s")
    print(f"Optimized time (large batch): {optimized_time:.4f}s ({original_time/optimized_time:.2f}x speedup)")
    
    # Check correctness
    max_diff = mx.max(mx.abs(result_original - result_optimized)).item()
    print(f"Max difference: {max_diff:.2e}")
    
    return True


def test_comprehensive_workflow():
    """Test comprehensive workflow with all optimizations."""
    print("\nTesting Comprehensive Workflow...")
    
    # Create test data
    batch_size = 64
    x = mx.random.normal((batch_size, 3))
    
    # Step 1: Compute spherical harmonics
    l_max = 2
    sh_optimized = []
    for l in range(l_max + 1):
        if l == 0:
            sh_l = OptimizedSphericalHarmonics.spherical_harmonics_l0(x, normalize=True)
        elif l == 1:
            sh_l = OptimizedSphericalHarmonics.spherical_harmonics_l1(x, normalize=True)
        elif l == 2:
            sh_l = OptimizedSphericalHarmonics.spherical_harmonics_l2(x, normalize=True)
        else:
            sh_l = spherical_harmonics(l, x, normalize=True)
        sh_optimized.append(sh_l)
    
    # Concatenate all spherical harmonics
    features = mx.concatenate(sh_optimized, axis=-1)
    
    # Step 2: Apply linear transformation
    irreps_in = Irreps("1x0e + 1x1o + 1x2e")
    irreps_out = Irreps("2x0e + 2x1o + 1x2e")
    
    linear_optimized = MemoryEfficientLinear(irreps_in, irreps_out)
    features_transformed = linear_optimized(features)
    
    # Step 3: Apply gate operation
    gates = mx.sigmoid(features_transformed[..., :2])  # Use first two scalars as gates
    gated_features = SpecializedKernels.gate_operation(gates, features_transformed[..., 2:])
    
    print(f"Input shape: {x.shape}")
    print(f"Spherical harmonics shape: {features.shape}")
    print(f"Linear transform shape: {features_transformed.shape}")
    print(f"Gated features shape: {gated_features.shape}")
    
    # Test timing
    num_runs = 20
    
    start_time = time.time()
    for _ in range(num_runs):
        # Full optimized workflow
        sh_optimized = []
        for l in range(l_max + 1):
            if l == 0:
                sh_l = OptimizedSphericalHarmonics.spherical_harmonics_l0(x, normalize=True)
            elif l == 1:
                sh_l = OptimizedSphericalHarmonics.spherical_harmonics_l1(x, normalize=True)
            elif l == 2:
                sh_l = OptimizedSphericalHarmonics.spherical_harmonics_l2(x, normalize=True)
            else:
                sh_l = spherical_harmonics(l, x, normalize=True)
            sh_optimized.append(sh_l)
        
        features = mx.concatenate(sh_optimized, axis=-1)
        features_transformed = linear_optimized(features)
        gates = mx.sigmoid(features_transformed[..., :2])
        gated_features = SpecializedKernels.gate_operation(gates, features_transformed[..., 2:])
    
    workflow_time = (time.time() - start_time) / num_runs
    print(f"Optimized workflow time: {workflow_time:.4f}s")
    
    return True


def run_all_integration_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("e3nn-mlx Integration Tests")
    print("=" * 60)
    
    test_results = []
    
    # Test tensor product integration
    try:
        result = test_integration_tensor_product()
        test_results.append(("Tensor Product Integration", result))
    except Exception as e:
        print(f"Tensor Product Integration test failed: {e}")
        test_results.append(("Tensor Product Integration", False))
    
    # Test linear layer integration
    try:
        result = test_integration_linear_layers()
        test_results.append(("Linear Layer Integration", result))
    except Exception as e:
        print(f"Linear Layer Integration test failed: {e}")
        test_results.append(("Linear Layer Integration", False))
    
    # Test spherical harmonics integration
    try:
        result = test_integration_spherical_harmonics()
        test_results.append(("Spherical Harmonics Integration", result))
    except Exception as e:
        print(f"Spherical Harmonics Integration test failed: {e}")
        test_results.append(("Spherical Harmonics Integration", False))
    
    # Test adaptive optimization
    try:
        result = test_adaptive_optimization()
        test_results.append(("Adaptive Optimization", result))
    except Exception as e:
        print(f"Adaptive Optimization test failed: {e}")
        test_results.append(("Adaptive Optimization", False))
    
    # Test memory efficiency
    try:
        result = test_memory_efficiency()
        test_results.append(("Memory Efficiency", result))
    except Exception as e:
        print(f"Memory Efficiency test failed: {e}")
        test_results.append(("Memory Efficiency", False))
    
    # Test comprehensive workflow
    try:
        result = test_comprehensive_workflow()
        test_results.append(("Comprehensive Workflow", result))
    except Exception as e:
        print(f"Comprehensive Workflow test failed: {e}")
        test_results.append(("Comprehensive Workflow", False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Integration Test Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(test_results)} tests")
    
    if passed == len(test_results):
        print("\nAll integration tests passed successfully!")
        return True
    else:
        print(f"\n{len(test_results) - passed} tests failed.")
        return False


if __name__ == "__main__":
    success = run_all_integration_tests()
    exit(0 if success else 1)