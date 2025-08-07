import e3nn_mlx as e3nn


def test_opt_defaults() -> None:
    """Test optimization defaults for MLX tensor products and linear layers."""
    from e3nn_mlx.o3 import TensorProduct, Linear
    
    # Test that we can create basic operations
    irreps_in1 = e3nn.o3.Irreps("2x1e")
    irreps_in2 = e3nn.o3.Irreps("2x1e")
    irreps_out = e3nn.o3.Irreps("2x0e")
    
    instructions = [
        (0, 0, 0, "uvw", True),
    ]
    
    # Test basic creation
    a = TensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        internal_weights=True
    )
    
    b = Linear("4x1o", "4x1o")
    
    # Test that objects are created successfully
    assert a is not None
    assert b is not None
    
    # Test optimization defaults can be set and retrieved
    old_defaults = e3nn.get_optimization_defaults()
    try:
        e3nn.set_optimization_defaults(optimize_einsums=False)
        new_defaults = e3nn.get_optimization_defaults()
        assert new_defaults["optimize_einsums"] is False
    finally:
        e3nn.set_optimization_defaults(**old_defaults)
    
    # Verify defaults are restored
    final_defaults = e3nn.get_optimization_defaults()
    assert final_defaults["optimize_einsums"] is True