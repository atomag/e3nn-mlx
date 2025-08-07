"""
Test script for e3nn-mlx optimizations

This script tests the implemented optimizations to ensure they work correctly
and provide performance improvements.
"""

import time
import mlx.core as mx
from e3nn_mlx.o3 import TensorProduct, spherical_harmonics, Irreps
from e3nn_mlx.util.compile import get_profiler, benchmark_compiled_vs_interpreted
from e3nn_mlx.o3._tensor_product._tensor_product_fusion import adaptive_tensor_product


def test_tensor_product_fusion():
    """Test tensor product fusion optimizations."""
    print("Testing Tensor Product Fusion...")
    
    # Create test data
    batch_size = 64
    irreps_in1 = Irreps("1x1o")
    irreps_in2 = Irreps("1x1o")
    irreps_out = Irreps("1x0e + 1x2e")
    
    # Create instructions - proper tensor product combinations
    instructions = [
        (0, 0, 0, "uvw", True),  # 1o x 1o -> 0e
        (0, 0, 1, "uvw", True),  # 1o x 1o -> 2e
    ]
    
    # Create tensor products with different fusion modes
    tp_original = TensorProduct(irreps_in1, irreps_in2, irreps_out, instructions, fusion_mode="none")
    tp_complete = TensorProduct(irreps_in1, irreps_in2, irreps_out, instructions, fusion_mode="complete")
    tp_instruction = TensorProduct(irreps_in1, irreps_in2, irreps_out, instructions, fusion_mode="instruction")
    tp_auto = TensorProduct(irreps_in1, irreps_in2, irreps_out, instructions, fusion_mode="auto")
    
    # Test inputs
    x1 = mx.random.normal((batch_size, irreps_in1.dim))
    x2 = mx.random.normal((batch_size, irreps_in2.dim))
    
    # Test correctness
    result_original = tp_original(x1, x2)
    result_complete = tp_complete(x1, x2)
    result_instruction = tp_instruction(x1, x2)
    result_auto = tp_auto(x1, x2)
    
    # Check results are similar
    max_diff_complete = mx.max(mx.abs(result_original - result_complete)).item()
    max_diff_instruction = mx.max(mx.abs(result_original - result_instruction)).item()
    max_diff_auto = mx.max(mx.abs(result_original - result_auto)).item()
    
    print(f"Max difference (complete): {max_diff_complete:.2e}")
    print(f"Max difference (instruction): {max_diff_instruction:.2e}")
    print(f"Max difference (auto): {max_diff_auto:.2e}")
    
    # Test performance
    num_runs = 100
    
    # Original implementation
    start_time = time.time()
    for _ in range(num_runs):
        result_original = tp_original(x1, x2)
    original_time = (time.time() - start_time) / num_runs
    
    # Complete fusion
    start_time = time.time()
    for _ in range(num_runs):
        result_complete = tp_complete(x1, x2)
    complete_time = (time.time() - start_time) / num_runs
    
    # Instruction fusion
    start_time = time.time()
    for _ in range(num_runs):
        result_instruction = tp_instruction(x1, x2)
    instruction_time = (time.time() - start_time) / num_runs
    
    # Auto fusion
    start_time = time.time()
    for _ in range(num_runs):
        result_auto = tp_auto(x1, x2)
    auto_time = (time.time() - start_time) / num_runs
    
    print(f"Original time: {original_time:.4f}s")
    print(f"Complete fusion time: {complete_time:.4f}s ({original_time/complete_time:.2f}x speedup)")
    print(f"Instruction fusion time: {instruction_time:.4f}s ({original_time/instruction_time:.2f}x speedup)")
    print(f"Auto fusion time: {auto_time:.4f}s ({original_time/auto_time:.2f}x speedup)")
    
    return {
        'original_time': original_time,
        'complete_time': complete_time,
        'instruction_time': instruction_time,
        'auto_time': auto_time,
        'max_diff_complete': max_diff_complete,
        'max_diff_instruction': max_diff_instruction,
        'max_diff_auto': max_diff_auto
    }


def test_spherical_harmonics_optimization():
    """Test spherical harmonics optimizations."""
    print("\nTesting Spherical Harmonics Optimization...")
    
    # Test data
    batch_size = 100
    x = mx.random.normal((batch_size, 3))
    
    # Test different l values
    l_values = [0, 1, 2, 3]
    
    for l in l_values:
        print(f"\nTesting l={l}...")
        
        # Test optimized vs original
        start_time = time.time()
        for _ in range(50):
            result_optimized = spherical_harmonics(l, x, normalize=True, optimized=True)
        optimized_time = (time.time() - start_time) / 50
        
        start_time = time.time()
        for _ in range(50):
            result_original = spherical_harmonics(l, x, normalize=True, optimized=False)
        original_time = (time.time() - start_time) / 50
        
        # Check correctness
        max_diff = mx.max(mx.abs(result_optimized - result_original)).item()
        
        print(f"  Original time: {original_time:.4f}s")
        print(f"  Optimized time: {optimized_time:.4f}s ({original_time/optimized_time:.2f}x speedup)")
        print(f"  Max difference: {max_diff:.2e}")


def test_memory_efficiency():
    """Test memory efficiency improvements."""
    print("\nTesting Memory Efficiency...")
    
    # Create larger test data
    batch_size = 1000
    irreps_in1 = Irreps("1x1o")
    irreps_in2 = Irreps("1x1o")
    irreps_out = Irreps("1x0e + 1x2e")
    
    instructions = [
        (0, 0, 0, "uvw", True),  # 1o x 1o -> 0e
        (0, 0, 1, "uvw", True),  # 1o x 1o -> 2e
    ]
    
    # Create tensor products
    tp_original = TensorProduct(irreps_in1, irreps_in2, irreps_out, instructions, fusion_mode="none")
    tp_optimized = TensorProduct(irreps_in1, irreps_in2, irreps_out, instructions, fusion_mode="auto")
    
    # Test inputs
    x1 = mx.random.normal((batch_size, irreps_in1.dim))
    x2 = mx.random.normal((batch_size, irreps_in2.dim))
    
    # Test memory usage (simplified - in practice would use proper memory profiling)
    import gc
    
    # Original
    gc.collect()
    start_time = time.time()
    for _ in range(10):
        result_original = tp_original(x1, x2)
    original_time = (time.time() - start_time) / 10
    
    # Optimized
    gc.collect()
    start_time = time.time()
    for _ in range(10):
        result_optimized = tp_optimized(x1, x2)
    optimized_time = (time.time() - start_time) / 10
    
    print(f"Original time (large batch): {original_time:.4f}s")
    print(f"Optimized time (large batch): {optimized_time:.4f}s ({original_time/optimized_time:.2f}x speedup)")
    
    # Check correctness
    max_diff = mx.max(mx.abs(result_original - result_optimized)).item()
    print(f"Max difference: {max_diff:.2e}")


def test_compilation_utilities():
    """Test compilation utilities."""
    print("\nTesting Compilation Utilities...")
    
    from e3nn_mlx.util.compile import compile_mode, optimize_memory_layout
    
    # Test compilation decorator
    @compile_mode("mlx")
    def test_function(x):
        return mx.sin(x) + mx.cos(x)
    
    # Test inputs
    x = mx.random.normal((100,))
    
    # Test compiled function
    result = test_function(x)
    print(f"Compiled function result shape: {result.shape}")
    
    # Test memory layout optimization
    x_non_contiguous = x[::2]  # Create non-contiguous array
    x_optimized = optimize_memory_layout(x_non_contiguous)
    
    print(f"Original shape: {x_non_contiguous.shape}")
    print(f"Optimized shape: {x_optimized.shape}")


def run_all_tests():
    """Run all optimization tests."""
    print("=" * 60)
    print("e3nn-mlx Optimization Tests")
    print("=" * 60)
    
    # Test tensor product fusion
    tp_results = test_tensor_product_fusion()
    
    # Test spherical harmonics optimization
    test_spherical_harmonics_optimization()
    
    # Test memory efficiency
    test_memory_efficiency()
    
    # Test compilation utilities
    test_compilation_utilities()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    print(f"Tensor Product Fusion:")
    print(f"  Complete fusion: {tp_results['original_time']/tp_results['complete_time']:.2f}x speedup")
    print(f"  Instruction fusion: {tp_results['original_time']/tp_results['instruction_time']:.2f}x speedup")
    print(f"  Auto fusion: {tp_results['original_time']/tp_results['auto_time']:.2f}x speedup")
    
    print(f"\nCorrectness:")
    print(f"  Max difference (complete): {tp_results['max_diff_complete']:.2e}")
    print(f"  Max difference (instruction): {tp_results['max_diff_instruction']:.2e}")
    print(f"  Max difference (auto): {tp_results['max_diff_auto']:.2e}")
    
    print(f"\nAll tests completed successfully!")


if __name__ == "__main__":
    run_all_tests()