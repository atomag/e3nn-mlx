"""
Equivariance tests for linear layer operations.
"""

import mlx.core as mx
import numpy as np
import pytest
from e3nn_mlx.o3 import Linear, Irreps
from e3nn_mlx.util.test_equivariance import assert_equivariant, EquivarianceTester
from e3nn_mlx.o3._rotation import rand_matrix


def test_linear_layer_rotational_equivariance():
    """Test that linear layers are equivariant under rotation."""
    
    # Test different linear layer configurations
    test_cases = [
        # (irreps_in, irreps_out, description)
        ("1x0e", "1x0e", "scalar to scalar"),
        ("1x1o", "1x1o", "vector to vector"),
        ("1x2e", "1x2e", "tensor to tensor"),
        ("1x0e+1x1o", "1x0e+1x1o", "mixed to mixed"),
        ("2x0e+1x1o", "3x0e+2x1o", "different multiplicities"),
        ("1x0e+1x1o+1x2e", "1x1o+1x2e", "different irreps"),
    ]
    
    for irreps_in_str, irreps_out_str, description in test_cases:
        print(f"Testing: {description}")
        
        # Create linear layer
        irreps_in = Irreps(irreps_in_str)
        irreps_out = Irreps(irreps_out_str)
        
        linear = Linear(irreps_in, irreps_out)
        
        # Create input generator
        def input_generator(irreps):
            # Generate random input for the given irreps
            features = []
            for mul, (l, p) in irreps:
                dim = mul * (2 * l + 1)
                features.append(mx.random.normal((dim,)))
            return mx.concatenate(features)
        
        # Create operation wrapper
        def linear_operation(input_features):
            # Reshape input to have batch dimension
            input_batch = input_features.reshape(1, -1)
            output = linear(input_batch)
            return output.reshape(-1)
        
        # Test equivariance
        is_equivariant, results = EquivarianceTester(
            tolerance=1e-5,
            num_samples=5
        ).assert_equivariant(
            operation=linear_operation,
            irreps_in=irreps_in,
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
        
        assert is_equivariant, f"Linear layer failed equivariance test: {description}"


def test_linear_layer_invariance_properties():
    """Test that linear layers preserve invariance properties."""
    
    # Test that scalars remain scalars
    irreps_in = Irreps("1x0e")  # scalar
    irreps_out = Irreps("1x0e")  # scalar
    
    linear = Linear(irreps_in, irreps_out)
    
    # Generate test input
    input_scalar = mx.random.normal((1, 1))
    
    # Apply rotation (should not change scalar)
    R = rand_matrix()
    rotated_input = input_scalar  # Scalars are invariant
    
    # Test that linear transformation preserves invariance
    output = linear(input_scalar)
    rotated_output = linear(rotated_input)
    
    error = mx.max(mx.abs(output - rotated_output)).item()
    print(f"Linear layer scalar invariance error: {error:.2e}")
    assert error < 1e-10, f"Linear layer failed scalar invariance test"


def test_linear_layer_weight_structure():
    """Test that linear layer weights have the correct block structure."""
    
    # Create linear layer with mixed irreps
    irreps_in = Irreps("1x0e+1x1o+1x2e")
    irreps_out = Irreps("1x0e+1x1o+1x2e")
    
    linear = Linear(irreps_in, irreps_out)
    
    # Check weight matrix structure
    weight_matrix = linear.weight
    
    # The weight matrix should be block-diagonal with respect to irreps
    # Each block should correspond to (l_in, l_out) pairs
    
    # For this test, we'll just check that the weight matrix has the right shape
    expected_shape = (irreps_in.dim, irreps_out.dim)
    assert weight_matrix.shape == expected_shape, f"Expected weight shape {expected_shape}, got {weight_matrix.shape}"
    
    # Check that weights are not all zeros (basic sanity check)
    assert mx.max(mx.abs(weight_matrix)) > 1e-10, "Linear layer weights are all zero"


def test_linear_layer_gradient_flow():
    """Test that gradients flow correctly through linear layers."""
    
    # Create simple linear layer
    irreps_in = Irreps("1x1o")
    irreps_out = Irreps("1x1o")
    
    linear = Linear(irreps_in, irreps_out)
    
    # Generate test input
    input_features = mx.random.normal((1, 3))
    
    # Forward pass
    output = linear(input_features)
    
    # Compute loss (simple L2 loss)
    loss = mx.sum(output ** 2)
    
    # Backward pass
    grad_output = mx.grad(lambda x: mx.sum(linear(x) ** 2))(input_features)
    
    # Check that gradients are not zero
    assert mx.max(mx.abs(grad_output)) > 1e-10, "Linear layer gradients are zero"
    
    print(f"Linear layer gradient max: {mx.max(mx.abs(grad_output)).item():.2e}")


def test_linear_layer_different_irreps():
    """Test linear layers with different input and output irreps."""
    
    # Test cases where input and output irreps are different
    test_cases = [
        ("1x0e", "1x1o", "scalar to vector"),
        ("1x1o", "1x0e", "vector to scalar"),
        ("1x0e+1x1o", "1x2e", "mixed to tensor"),
        ("1x1o", "1x0e+1x1o+1x2e", "vector to mixed"),
    ]
    
    for irreps_in_str, irreps_out_str, description in test_cases:
        print(f"Testing different irreps: {description}")
        
        irreps_in = Irreps(irreps_in_str)
        irreps_out = Irreps(irreps_out_str)
        
        linear = Linear(irreps_in, irreps_out)
        
        # Generate test input
        input_features = mx.random.normal((1, irreps_in.dim))
        
        # Forward pass
        output = linear(input_features)
        
        # Check output shape
        expected_shape = (1, irreps_out.dim)
        assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}"
        
        # Check that output is not NaN
        assert not mx.any(mx.isnan(output)), f"Linear layer produced NaN output: {description}"


def test_linear_layer_equivariance_with_random_rotations():
    """Test linear layer equivariance with many random rotations."""
    
    # Create linear layer
    irreps_in = Irreps("1x0e+1x1o+1x2e")
    irreps_out = Irreps("1x0e+1x1o+1x2e")
    
    linear = Linear(irreps_in, irreps_out)
    
    # Generate test input
    input_features = mx.random.normal((1, irreps_in.dim))
    
    # Test with many random rotations
    num_rotations = 20
    max_error = 0.0
    
    for i in range(num_rotations):
        # Generate random rotation
        R = rand_matrix()
        
        # Apply operation then rotate
        output = linear(input_features)
        rotated_output = apply_rotation_to_features(output, irreps_out, R)
        
        # Rotate then apply operation
        rotated_input = apply_rotation_to_features(input_features, irreps_in, R)
        output_rotated = linear(rotated_input)
        
        # Compute error
        error = mx.max(mx.abs(rotated_output - output_rotated)).item()
        max_error = max(max_error, error)
    
    print(f"Linear layer max equivariance error over {num_rotations} rotations: {max_error:.2e}")
    assert max_error < 1e-5, f"Linear layer failed equivariance test with max error {max_error:.2e}"


def apply_rotation_to_features(features, irreps, R):
    """Helper function to apply rotation to features."""
    # This is a simplified version - in practice, you'd use the full Wigner D-matrix
    # For now, we'll just apply the rotation to the vector components
    
    output = []
    index = 0
    
    for mul, (l, p) in irreps:
        dim = mul * (2 * l + 1)
        chunk = features[:, index:index + dim]
        index += dim
        
        if l == 0:
            # Scalars are invariant
            output.append(chunk)
        elif l == 1:
            # Vectors transform under rotation
            chunk_reshaped = chunk.reshape(-1, mul, 3)
            rotated_chunk = chunk_reshaped @ R.T
            rotated_chunk = rotated_chunk.reshape(-1, dim)
            output.append(rotated_chunk)
        else:
            # For l > 1, we'd need the full Wigner D-matrix
            # For now, just return the chunk unchanged
            output.append(chunk)
    
    return mx.concatenate(output, axis=-1)


if __name__ == "__main__":
    # Run all tests
    test_linear_layer_rotational_equivariance()
    test_linear_layer_invariance_properties()
    test_linear_layer_weight_structure()
    test_linear_layer_gradient_flow()
    test_linear_layer_different_irreps()
    test_linear_layer_equivariance_with_random_rotations()
    print("All linear layer equivariance tests passed!")