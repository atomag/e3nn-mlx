#!/usr/bin/env python3
"""Test the MLX tensor product implementation."""

import mlx.core as mx
import numpy as np
from e3nn_mlx.o3._tensor_product import TensorProduct, FullyConnectedTensorProduct, ElementwiseTensorProduct
from e3nn_mlx.o3._irreps import Irreps


def test_basic_tensor_product():
    """Test basic tensor product functionality."""
    print("Testing basic tensor product...")
    
    # Create simple tensor product
    irreps_in1 = Irreps("2x0e + 1x1o")
    irreps_in2 = Irreps("1x0e + 1x1o")
    irreps_out = Irreps("2x0e + 3x1o + 1x2e")
    
    # Create instructions for allowed paths
    instructions = [
        (0, 0, 0, "uvw", True),  # 0e x 0e -> 0e
        (0, 1, 1, "uvw", True),  # 0e x 1o -> 1o
        (1, 0, 1, "uvw", True),  # 1o x 0e -> 1o
        (1, 1, 0, "uvw", True),  # 1o x 1o -> 0e
        (1, 1, 2, "uvw", True),  # 1o x 1o -> 2e
    ]
    
    tp = TensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        internal_weights=True
    )
    
    print(f"Created tensor product: {tp}")
    print(f"Weight numel: {tp.weight_numel}")
    
    # Test with random inputs
    batch_size = 3
    x1 = mx.random.normal((batch_size, irreps_in1.dim))
    x2 = mx.random.normal((batch_size, irreps_in2.dim))
    
    output = tp(x1, x2)
    print(f"Input shapes: {x1.shape}, {x2.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output dim: {irreps_out.dim}")
    
    assert output.shape == (batch_size, irreps_out.dim), f"Expected {(batch_size, irreps_out.dim)}, got {output.shape}"
    print("✓ Basic tensor product test passed")


def test_fully_connected_tensor_product():
    """Test fully connected tensor product."""
    print("\nTesting fully connected tensor product...")
    
    irreps_in1 = Irreps("2x0e + 1x1o")
    irreps_in2 = Irreps("1x0e + 1x1o")
    irreps_out = Irreps("2x0e + 3x1o + 1x2e")
    
    tp = FullyConnectedTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        internal_weights=True
    )
    
    print(f"Created FC tensor product: {tp}")
    
    # Test with random inputs
    batch_size = 2
    x1 = mx.random.normal((batch_size, irreps_in1.dim))
    x2 = mx.random.normal((batch_size, irreps_in2.dim))
    
    output = tp(x1, x2)
    print(f"Input shapes: {x1.shape}, {x2.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (batch_size, irreps_out.dim)
    print("✓ Fully connected tensor product test passed")


def test_elementwise_tensor_product():
    """Test elementwise tensor product."""
    print("\nTesting elementwise tensor product...")
    
    irreps_in1 = Irreps("3x1o")
    irreps_in2 = Irreps("3x1o")
    
    tp = ElementwiseTensorProduct(
        irreps_in1,
        irreps_in2,
        internal_weights=False  # No weights for elementwise
    )
    
    print(f"Created elementwise tensor product: {tp}")
    
    # Test with random inputs
    batch_size = 4
    x1 = mx.random.normal((batch_size, irreps_in1.dim))
    x2 = mx.random.normal((batch_size, irreps_in2.dim))
    
    output = tp(x1, x2)
    print(f"Input shapes: {x1.shape}, {x2.shape}")
    print(f"Output shape: {output.shape}")
    
    print("✓ Elementwise tensor product test passed")


def test_right_method():
    """Test the right method for partial evaluation."""
    print("\nTesting right method...")
    
    irreps_in1 = Irreps("1x1o")
    irreps_in2 = Irreps("1x1o")
    irreps_out = Irreps("1x1e")
    
    instructions = [(0, 0, 0, "uvw", True)]
    
    tp = TensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        internal_weights=True
    )
    
    # Test right method
    batch_size = 2
    y = mx.random.normal((batch_size, irreps_in2.dim))
    
    right_result = tp.right(y)
    print(f"Right method output shape: {right_result.shape}")
    print(f"Expected shape: {y.shape[:-1] + (irreps_in1.dim, irreps_out.dim)}")
    
    expected_shape = y.shape[:-1] + (irreps_in1.dim, irreps_out.dim)
    assert right_result.shape == expected_shape
    print("✓ Right method test passed")


if __name__ == "__main__":
    print("Running MLX Tensor Product Tests")
    print("=" * 50)
    
    try:
        test_basic_tensor_product()
        test_fully_connected_tensor_product()
        test_elementwise_tensor_product()
        test_right_method()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()