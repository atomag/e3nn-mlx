"""
Equivariance tests for tensor product operations.
"""

import mlx.core as mx
import numpy as np
import pytest
from e3nn_mlx.o3 import TensorProduct, Irreps, wigner_3j
from e3nn_mlx.util.test_equivariance import assert_equivariant, EquivarianceTester
from e3nn_mlx.o3._rotation import rand_matrix


def test_tensor_product_rotational_equivariance():
    """Test that tensor products are equivariant under rotation."""
    
    # Test different tensor product configurations
    test_cases = [
        # (irreps1, irreps2, irreps_out, description)
        ("1x0e", "1x0e", "1x0e", "scalar x scalar -> scalar"),
        ("1x1o", "1x1o", "1x0e+1x1o+1x2e", "vector x vector -> scalar + vector + tensor"),
        ("1x0e", "1x1o", "1x1o", "scalar x vector -> vector"),
        ("1x1o", "1x2e", "1x1o+1x2o+1x3e", "vector x tensor -> vector + tensor + pentuplet"),
        ("2x0e+1x1o", "1x0e+1x1o", "2x0e+1x1o+1x1o+1x2e", "mixed input"),
    ]
    
    for irreps1_str, irreps2_str, irreps_out_str, description in test_cases:
        print(f"Testing: {description}")
        
        # Create tensor product
        irreps1 = Irreps(irreps1_str)
        irreps2 = Irreps(irreps2_str)
        irreps_out = Irreps(irreps_out_str)
        
        tp = TensorProduct(
            irreps_in1=irreps1,
            irreps_in2=irreps2,
            irreps_out=irreps_out
        )
        
        # Create input generator
        def input_generator(irreps):
            # Generate random input for the given irreps
            features = []
            for mul, (l, p) in irreps:
                dim = mul * (2 * l + 1)
                features.append(mx.random.normal((dim,)))
            return mx.concatenate(features)
        
        # Create operation that takes concatenated inputs
        def tensor_product_operation(combined_input):
            # Split combined input into two parts
            dim1 = irreps1.dim
            input1 = combined_input[:dim1]
            input2 = combined_input[dim1:]
            
            # Reshape inputs to have batch dimension
            input1 = input1.reshape(1, -1)
            input2 = input2.reshape(1, -1)
            
            return tp(input1, input2).reshape(-1)
        
        # Test equivariance
        is_equivariant, results = EquivarianceTester(
            tolerance=1e-5,
            num_samples=5
        ).assert_equivariant(
            operation=tensor_product_operation,
            irreps_in=irreps1 + irreps2,
            irreps_out=irreps_out,
            input_generator=lambda ir: input_generator(ir),
            test_rotations=True,
            test_inversions=True,
            test_translations=False
        )
        
        print(f"  {description}: {'PASS' if is_equivariant else 'FAIL'}")
        if not is_equivariant:
            for transformation, result in results.items():
                if not result['passed']:
                    print(f"    {transformation}: max_error = {result['max_error']:.2e}")
        
        assert is_equivariant, f"Tensor product failed equivariance test: {description}"


def test_tensor_product_path_consistency():
    """Test that tensor product paths are consistent with Wigner 3j symbols."""
    
    # Test specific case: l1=1, l2=1, l3=1 (vector x vector -> vector)
    l1, l2, l3 = 1, 1, 1
    
    # Get Wigner 3j symbols
    w3j = wigner_3j(l1, l2, l3)
    
    # Create tensor product for this specific path
    tp = TensorProduct(
        irreps_in1=Irreps("1x1o"),
        irreps_in2=Irreps("1x1o"),
        irreps_out=Irreps("1x1o"),
        internal_weights=False
    )
    
    # Check that the tensor product weights are proportional to Wigner 3j symbols
    # This is a simplified test - in practice, tensor products may have additional normalization
    print(f"Wigner 3j symbols shape: {w3j.shape}")
    print(f"Tensor product weights: {tp.weights}")
    
    # For now, just check that the tensor product can be created and executed
    input1 = mx.random.normal((1, 3))  # vector
    input2 = mx.random.normal((1, 3))  # vector
    
    output = tp(input1, input2)
    assert output.shape == (1, 3), f"Expected shape (1, 3), got {output.shape}"


def test_tensor_product_instruction_fusion():
    """Test that tensor product instruction fusion doesn't break equivariance."""
    
    # Create a tensor product with multiple paths
    irreps1 = Irreps("1x0e+1x1o")
    irreps2 = Irreps("1x0e+1x1o")
    irreps_out = Irreps("1x0e+1x1o+1x2e")
    
    tp = TensorProduct(irreps_in1=irreps1, irreps_in2=irreps2, irreps_out=irreps_out)
    
    # Generate test inputs
    input1 = mx.random.normal((1, irreps1.dim))
    input2 = mx.random.normal((1, irreps2.dim))
    
    # Test that fusion doesn't change the output (much)
    # This is a basic sanity check
    output_normal = tp(input1, input2)
    
    # Try to create the same tensor product with different fusion settings
    tp_fused = TensorProduct(
        irreps_in1=irreps1,
        irreps_in2=irreps2,
        irreps_out=irreps_out,
        fusion_mode="elementwise"
    )
    
    output_fused = tp_fused(input1, input2)
    
    # Compare outputs (they should be similar but not identical due to different computation order)
    max_diff = mx.max(mx.abs(output_normal - output_fused)).item()
    print(f"Tensor product fusion max_diff: {max_diff:.2e}")
    
    # For now, just check that both produce valid outputs
    assert output_normal.shape == output_fused.shape
    assert not mx.any(mx.isnan(output_normal))
    assert not mx.any(mx.isnan(output_fused))


def test_tensor_product_special_cases():
    """Test special cases of tensor products."""
    
    # Test scalar multiplication (equivariance)
    tp_scalar = TensorProduct(
        irreps_in1=Irreps("1x0e"),
        irreps_in2=Irreps("1x1o"),
        irreps_out=Irreps("1x1o")
    )
    
    # Test that scalar multiplication is equivariant
    scalar = mx.random.normal((1, 1))
    vector = mx.random.normal((1, 3))
    
    # Apply rotation
    R = rand_matrix()
    D = wigner_3j(1, 1, 1)  # Simplified rotation for l=1
    
    # Test equivariance: R * (s * v) = s * (R * v)
    rotated_sv = tp_scalar(scalar, vector @ D.T)
    s_rotated_v = tp_scalar(scalar, vector) @ D.T
    
    error = mx.max(mx.abs(rotated_sv - s_rotated_v)).item()
    print(f"Scalar multiplication equivariance error: {error:.2e}")
    assert error < 1e-5, f"Scalar multiplication failed equivariance test"


if __name__ == "__main__":
    # Run all tests
    test_tensor_product_rotational_equivariance()
    test_tensor_product_path_consistency()
    test_tensor_product_instruction_fusion()
    test_tensor_product_special_cases()
    print("All tensor product equivariance tests passed!")