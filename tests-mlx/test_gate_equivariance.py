"""
Equivariance tests for gate activation operations.
"""

import mlx.core as mx
import numpy as np
import pytest
from e3nn_mlx.nn import Gate
from e3nn_mlx.o3 import Irreps
from e3nn_mlx.util.test_equivariance import assert_equivariant, EquivarianceTester
from e3nn_mlx.o3._rotation import rand_matrix


def test_gate_rotational_equivariance():
    """Test that gate activations are equivariant under rotation."""
    
    # Test different gate configurations
    test_cases = [
        # (irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated, description)
        ("1x0e", [mx.sigmoid], "1x0e", [mx.tanh], "1x1o", "simple scalar gate"),
        ("2x0e", [mx.sigmoid, mx.tanh], "2x0e", [mx.tanh, mx.sigmoid], "2x1o", "multiple gates"),
        ("1x0e", [mx.abs], "1x0e", [lambda x: x**2], "1x1o+1x2e", "mixed gated features"),
        ("1x0e+1x0o", [mx.sigmoid, None], "1x0e", [mx.tanh], "1x1o", "mixed scalar types"),
    ]
    
    for irreps_scalars_str, act_scalars, irreps_gates_str, act_gates, irreps_gated_str, description in test_cases:
        print(f"Testing: {description}")
        
        # Create gate
        irreps_scalars = Irreps(irreps_scalars_str)
        irreps_gates = Irreps(irreps_gates_str)
        irreps_gated = Irreps(irreps_gated_str)
        
        gate = Gate(irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated)
        
        # Create input generator
        def input_generator(irreps):
            # Generate random input for the given irreps
            features = []
            for mul, (l, p) in irreps:
                dim = mul * (2 * l + 1)
                features.append(mx.random.normal((dim,)))
            return mx.concatenate(features)
        
        # Create operation wrapper
        def gate_operation(input_features):
            return gate(input_features)
        
        # Test equivariance
        is_equivariant, results = EquivarianceTester(
            tolerance=1e-5,
            num_samples=5
        ).assert_equivariant(
            operation=gate_operation,
            irreps_in=gate.irreps_in,
            irreps_out=gate.irreps_out,
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
        
        assert is_equivariant, f"Gate failed equivariance test: {description}"


def test_gate_input_validation():
    """Test that gate validates input correctly."""
    
    # Test that non-scalar gates raise an error
    try:
        gate = Gate("1x0e", [mx.sigmoid], "1x1o", [mx.tanh], "1x1o")
        assert False, "Gate should have raised an error for non-scalar gates"
    except ValueError as e:
        print(f"Correctly caught error for non-scalar gates: {e}")
    
    # Test that mismatched gate/gated dimensions raise an error
    try:
        gate = Gate("1x0e", [mx.sigmoid], "2x0e", [mx.tanh], "1x1o")
        assert False, "Gate should have raised an error for mismatched dimensions"
    except ValueError as e:
        print(f"Correctly caught error for mismatched dimensions: {e}")


def test_gate_structure():
    """Test that gate has the correct internal structure."""
    
    # Create a simple gate
    irreps_scalars = Irreps("1x0e")
    irreps_gates = Irreps("1x0e")
    irreps_gated = Irreps("1x1o")
    
    gate = Gate(irreps_scalars, [mx.sigmoid], irreps_gates, [mx.tanh], irreps_gated)
    
    # Check that gate has the correct components
    assert hasattr(gate, 'sc'), "Gate should have shortcut component"
    assert hasattr(gate, 'act_scalars'), "Gate should have scalar activation"
    assert hasattr(gate, 'act_gates'), "Gate should have gate activation"
    assert hasattr(gate, 'mul'), "Gate should have multiplication component"
    
    # Check input/output irreps
    expected_input_dim = irreps_scalars.dim + irreps_gates.dim + irreps_gated.dim
    assert gate.irreps_in.dim == expected_input_dim, \
        f"Expected input dim {expected_input_dim}, got {gate.irreps_in.dim}"
    
    expected_output_dim = irreps_scalars.dim + irreps_gated.dim
    assert gate.irreps_out.dim == expected_output_dim, \
        f"Expected output dim {expected_output_dim}, got {gate.irreps_out.dim}"


def test_gate_forward_pass():
    """Test that gate forward pass works correctly."""
    
    # Create a simple gate
    irreps_scalars = Irreps("1x0e")
    irreps_gates = Irreps("1x0e")
    irreps_gated = Irreps("1x1o")
    
    gate = Gate(irreps_scalars, [mx.sigmoid], irreps_gates, [mx.tanh], irreps_gated)
    
    # Generate test input
    input_dim = gate.irreps_in.dim
    input_features = mx.random.normal((1, input_dim))
    
    # Forward pass
    output = gate(input_features)
    
    # Check output shape
    expected_shape = (1, gate.irreps_out.dim)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}"
    
    # Check that output is not NaN
    assert not mx.any(mx.isnan(output)), "Gate produced NaN output"


def test_gate_signal_flow():
    """Test that signal flows correctly through gate components."""
    
    # Create a gate with known structure
    irreps_scalars = Irreps("1x0e")
    irreps_gates = Irreps("1x0e")
    irreps_gated = Irreps("1x1o")
    
    gate = Gate(irreps_scalars, [mx.sigmoid], irreps_gates, [mx.tanh], irreps_gated)
    
    # Create test input with known values
    input_features = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])  # scalar, gate, gated
    
    # Forward pass
    output = gate(input_features)
    
    # Manually compute expected output
    # Input structure: [scalar, gate, gated_x, gated_y, gated_z]
    scalar_input = input_features[:, 0]
    gate_input = input_features[:, 1]
    gated_input = input_features[:, 2:5]
    
    # Expected computation
    activated_scalar = mx.sigmoid(scalar_input)
    activated_gate = mx.tanh(gate_input)
    gated_output = gated_input * activated_gate
    
    expected_output = mx.concatenate([activated_scalar, gated_output], axis=1)
    
    # Check that outputs match
    output_error = mx.max(mx.abs(output - expected_output)).item()
    print(f"Gate signal flow error: {output_error:.2e}")
    assert output_error < 1e-10, "Gate signal flow failed"


def test_gate_zero_input():
    """Test gate behavior with zero input."""
    
    # Create a simple gate
    irreps_scalars = Irreps("1x0e")
    irreps_gates = Irreps("1x0e")
    irreps_gated = Irreps("1x1o")
    
    gate = Gate(irreps_scalars, [mx.sigmoid], irreps_gates, [mx.tanh], irreps_gated)
    
    # Zero input
    input_dim = gate.irreps_in.dim
    zero_input = mx.zeros((1, input_dim))
    
    # Forward pass
    output = gate(zero_input)
    
    # Check that scalar part is sigmoid(0) = 0.5
    scalar_part = output[:, 0]
    expected_scalar = mx.sigmoid(mx.array([0.0]))
    scalar_error = mx.max(mx.abs(scalar_part - expected_scalar)).item()
    print(f"Gate zero input scalar error: {scalar_error:.2e}")
    assert scalar_error < 1e-10, "Gate zero input scalar failed"
    
    # Check that gated part is zero (since gate = tanh(0) = 0)
    gated_part = output[:, 1:4]
    gated_error = mx.max(mx.abs(gated_part)).item()
    print(f"Gate zero input gated error: {gated_error:.2e}")
    assert gated_error < 1e-10, "Gate zero input gated failed"


def test_gate_large_input():
    """Test gate behavior with large input values."""
    
    # Create a simple gate
    irreps_scalars = Irreps("1x0e")
    irreps_gates = Irreps("1x0e")
    irreps_gated = Irreps("1x1o")
    
    gate = Gate(irreps_scalars, [mx.sigmoid], irreps_gates, [mx.tanh], irreps_gated)
    
    # Large input
    input_dim = gate.irreps_in.dim
    large_input = mx.array([[100.0, -100.0, 10.0, -10.0, 5.0]])
    
    # Forward pass
    output = gate(large_input)
    
    # Check that scalar part is saturated sigmoid
    scalar_part = output[:, 0]
    expected_scalar = mx.sigmoid(mx.array([100.0]))
    scalar_error = mx.max(mx.abs(scalar_part - expected_scalar)).item()
    print(f"Gate large input scalar error: {scalar_error:.2e}")
    assert scalar_error < 1e-10, "Gate large input scalar failed"
    
    # Check that gate part is saturated tanh
    gate_part_manual = mx.tanh(mx.array([-100.0]))
    # We can't directly access the gate output, but we can check that the gated part is small
    gated_part = output[:, 1:4]
    gated_magnitude = mx.max(mx.abs(gated_part)).item()
    print(f"Gate large input gated magnitude: {gated_magnitude:.2e}")
    assert gated_magnitude < 1.0, "Gate large input gated failed"


def test_gate_gradient_flow():
    """Test that gradients flow correctly through gates."""
    
    # Create a simple gate
    irreps_scalars = Irreps("1x0e")
    irreps_gates = Irreps("1x0e")
    irreps_gated = Irreps("1x1o")
    
    gate = Gate(irreps_scalars, [mx.sigmoid], irreps_gates, [mx.tanh], irreps_gated)
    
    # Generate test input
    input_dim = gate.irreps_in.dim
    input_features = mx.random.normal((1, input_dim))
    
    # Compute loss and gradients
    def loss_fn(x):
        output = gate(x)
        return mx.sum(output ** 2)
    
    grad_input = mx.grad(loss_fn)(input_features)
    
    # Check that gradients are not zero
    assert mx.max(mx.abs(grad_input)) > 1e-10, "Gate gradients are zero"
    
    print(f"Gate gradient max: {mx.max(mx.abs(grad_input)).item():.2e}")


if __name__ == "__main__":
    # Run all tests
    test_gate_rotational_equivariance()
    test_gate_input_validation()
    test_gate_structure()
    test_gate_forward_pass()
    test_gate_signal_flow()
    test_gate_zero_input()
    test_gate_large_input()
    test_gate_gradient_flow()
    print("All gate activation equivariance tests passed!")