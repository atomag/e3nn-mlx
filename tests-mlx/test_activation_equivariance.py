"""
Equivariance tests for activation function operations.
"""

import mlx.core as mx
import numpy as np
import pytest
from e3nn_mlx.nn import Activation
from e3nn_mlx.o3 import Irreps
from e3nn_mlx.util.test_equivariance import assert_equivariant, EquivarianceTester
from e3nn_mlx.o3._rotation import rand_matrix


def test_activation_rotational_equivariance():
    """Test that activation functions are equivariant under rotation."""
    
    # Test different activation configurations
    test_cases = [
        # (irreps_in, acts, description)
        ("1x0e", [mx.abs], "scalar abs activation"),
        ("1x0e", [mx.tanh], "scalar tanh activation"),
        ("1x0e", [lambda x: x**2], "scalar square activation"),
        ("1x0e+1x1o", [mx.abs, None], "mixed activation"),
        ("2x0e+1x1o", [mx.tanh, None], "multiple scalars + vector"),
        ("1x0e+1x1o+1x2e", [None, None, None], "no activation"),
    ]
    
    for irreps_in_str, acts, description in test_cases:
        print(f"Testing: {description}")
        
        # Create activation
        irreps_in = Irreps(irreps_in_str)
        
        activation = Activation(irreps_in, acts)
        
        # Create input generator
        def input_generator(irreps):
            # Generate random input for the given irreps
            features = []
            for mul, (l, p) in irreps:
                dim = mul * (2 * l + 1)
                features.append(mx.random.normal((dim,)))
            return mx.concatenate(features)
        
        # Create operation wrapper
        def activation_operation(input_features):
            return activation(input_features)
        
        # Test equivariance
        is_equivariant, results = EquivarianceTester(
            tolerance=1e-5,
            num_samples=5
        ).assert_equivariant(
            operation=activation_operation,
            irreps_in=irreps_in,
            irreps_out=activation.irreps_out,
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
        
        assert is_equivariant, f"Activation failed equivariance test: {description}"


def test_activation_parity_preservation():
    """Test that activation functions preserve parity correctly."""
    
    # Test parity preservation for different activation functions
    test_cases = [
        # (irreps_in, acts, expected_irreps_out, description)
        ("1x0e", [mx.abs], "1x0e", "abs: odd scalar -> even scalar"),
        ("1x0o", [mx.abs], "1x0e", "abs: even scalar -> even scalar"),
        ("1x0e", [mx.tanh], "1x0e", "tanh: even scalar -> even scalar"),
        ("1x0o", [mx.tanh], "1x0o", "tanh: odd scalar -> odd scalar"),
        ("1x0e", [lambda x: x**3], "1x0e", "cube: even scalar -> even scalar"),
        ("1x0o", [lambda x: x**3], "1x0o", "cube: odd scalar -> odd scalar"),
        ("1x0e", [lambda x: x**2], "1x0e", "square: even scalar -> even scalar"),
        ("1x0o", [lambda x: x**2], "1x0e", "square: odd scalar -> even scalar"),
    ]
    
    for irreps_in_str, acts, expected_irreps_out_str, description in test_cases:
        print(f"Testing parity: {description}")
        
        irreps_in = Irreps(irreps_in_str)
        expected_irreps_out = Irreps(expected_irreps_out_str)
        
        activation = Activation(irreps_in, acts)
        
        # Check that output irreps match expected
        assert activation.irreps_out == expected_irreps_out, \
            f"Expected {expected_irreps_out}, got {activation.irreps_out}"
        
        # Test with actual values
        input_features = mx.random.normal((1, irreps_in.dim))
        output = activation(input_features)
        
        # Check output shape
        expected_shape = (1, expected_irreps_out.dim)
        assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}"


def test_activation_scalar_only():
    """Test that activation functions only apply to scalars."""
    
    # Test that non-scalar inputs raise an error
    irreps_in = Irreps("1x1o")  # vector (non-scalar)
    acts = [mx.abs]  # try to apply activation to vector
    
    # This should raise an error
    try:
        activation = Activation(irreps_in, acts)
        assert False, "Activation should have raised an error for non-scalar input"
    except ValueError as e:
        print(f"Correctly caught error: {e}")


def test_activation_normalization():
    """Test that activation functions are properly normalized."""
    
    # Test normalization property
    irreps_in = Irreps("1x0e")
    acts = [mx.tanh]
    
    activation = Activation(irreps_in, acts)
    
    # Generate test input
    input_features = mx.random.normal((1000, 1))  # many samples
    
    # Apply activation
    output = activation(input_features)
    
    # Check that output is normalized (second moment should be 1)
    second_moment = mx.mean(output ** 2).item()
    print(f"Activation normalization: second_moment = {second_moment:.6f}")
    
    # Should be close to 1
    assert abs(second_moment - 1.0) < 0.1, f"Activation not properly normalized: {second_moment}"


def test_activation_gate_like_behavior():
    """Test activation functions that behave like gates."""
    
    # Test gate-like activation: scalar activation that can gate other features
    irreps_in = Irreps("1x0e+1x1o")  # scalar + vector
    acts = [mx.sigmoid, None]  # activate scalar, leave vector unchanged
    
    activation = Activation(irreps_in, acts)
    
    # Generate test input
    input_features = mx.random.normal((1, irreps_in.dim))
    
    # Apply activation
    output = activation(input_features)
    
    # Check that scalar part is gated (between 0 and 1 for sigmoid)
    scalar_part = output[:, 0]  # first feature is the activated scalar
    assert mx.all(scalar_part >= 0) and mx.all(scalar_part <= 1), \
        f"Sigmoid activation failed: min={mx.min(scalar_part)}, max={mx.max(scalar_part)}"
    
    # Check that vector part is unchanged
    vector_part = output[:, 1:4]  # next 3 features are the vector
    input_vector = input_features[:, 1:4]
    vector_error = mx.max(mx.abs(vector_part - input_vector)).item()
    print(f"Vector preservation error: {vector_error:.2e}")
    assert vector_error < 1e-10, "Vector part was not preserved"


def test_activation_multiple_paths():
    """Test activation with multiple scalar paths."""
    
    # Test multiple scalar activations
    irreps_in = Irreps("2x0e+1x1o")  # 2 scalars + 1 vector
    acts = [mx.tanh, mx.abs, None]  # different activations for each scalar
    
    activation = Activation(irreps_in, acts)
    
    # Generate test input
    input_features = mx.random.normal((1, irreps_in.dim))
    
    # Apply activation
    output = activation(input_features)
    
    # Check output shape
    expected_shape = (1, activation.irreps_out.dim)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}"
    
    # Check that first scalar is tanh-transformed
    scalar1_input = input_features[:, 0]
    scalar1_output = output[:, 0]
    expected_scalar1 = mx.tanh(scalar1_input)
    scalar1_error = mx.max(mx.abs(scalar1_output - expected_scalar1)).item()
    print(f"First scalar activation error: {scalar1_error:.2e}")
    assert scalar1_error < 1e-10, "First scalar activation failed"
    
    # Check that second scalar is abs-transformed
    scalar2_input = input_features[:, 1]
    scalar2_output = output[:, 1]
    expected_scalar2 = mx.abs(scalar2_input)
    scalar2_error = mx.max(mx.abs(scalar2_output - expected_scalar2)).item()
    print(f"Second scalar activation error: {scalar2_error:.2e}")
    assert scalar2_error < 1e-10, "Second scalar activation failed"


def test_activation_inversion_behavior():
    """Test activation function behavior under inversion."""
    
    # Test that activation functions respect parity
    irreps_in = Irreps("1x0o")  # odd scalar
    acts = [mx.tanh]  # odd function
    
    activation = Activation(irreps_in, acts)
    
    # Generate test input
    input_features = mx.random.normal((1, 1))
    
    # Apply activation to input and inverted input
    output = activation(input_features)
    output_inverted = activation(-input_features)
    
    # For odd function + odd scalar: f(-x) = -f(x)
    expected_output_inverted = -output
    inversion_error = mx.max(mx.abs(output_inverted - expected_output_inverted)).item()
    print(f"Activation inversion error: {inversion_error:.2e}")
    assert inversion_error < 1e-10, "Activation inversion behavior failed"


if __name__ == "__main__":
    # Run all tests
    test_activation_rotational_equivariance()
    test_activation_parity_preservation()
    test_activation_scalar_only()
    test_activation_normalization()
    test_activation_gate_like_behavior()
    test_activation_multiple_paths()
    test_activation_inversion_behavior()
    print("All activation function equivariance tests passed!")