"""
Equivariance tests for spherical harmonics operations.
"""

import mlx.core as mx
import numpy as np
import pytest
from e3nn_mlx.o3 import spherical_harmonics, Irreps
from e3nn_mlx.util.test_equivariance import assert_equivariant, EquivarianceTester
from e3nn_mlx.o3._rotation import rand_matrix, angles_to_matrix


def test_spherical_harmonics_rotational_equivariance():
    """Test that spherical harmonics are equivariant under rotation."""
    def spherical_harmonics_wrapper(l_max, positions):
        """Wrapper for spherical harmonics that matches the expected interface."""
        # Generate spherical harmonics for all l up to l_max
        result = []
        for l in range(l_max + 1):
            Y = spherical_harmonics(l, positions, normalize=True)
            result.append(Y)
        return mx.concatenate(result, axis=-1)
    
    # Test for different l_max values
    for l_max in [1, 2, 3, 4]:
        # Generate random positions
        num_positions = 10
        positions = mx.random.normal((num_positions, 3))
        
        # Define irreps for spherical harmonics up to l_max
        irreps_out = []
        for l in range(l_max + 1):
            irreps_out.append((1, (l, (-1)**l)))  # Each l appears once
        irreps_out = Irreps(irreps_out)
        
        # Create input generator that returns positions
        def input_generator(irreps):
            # For spherical harmonics, input is just positions (3D vectors)
            return positions
        
        # Test equivariance
        is_equivariant, results = EquivarianceTester(
            tolerance=1e-5, 
            num_samples=5
        ).assert_equivariant(
            operation=lambda pos: spherical_harmonics_wrapper(l_max, pos),
            irreps_in="1x1o",  # Positions are vectors (l=1, odd parity)
            irreps_out=irreps_out,
            input_generator=input_generator,
            test_rotations=True,
            test_inversions=True,
            test_translations=False
        )
        
        print(f"Spherical harmonics l_max={l_max}: {'PASS' if is_equivariant else 'FAIL'}")
        if not is_equivariant:
            for transformation, result in results.items():
                if not result['passed']:
                    print(f"  {transformation}: max_error = {result['max_error']:.2e}")
        
        assert is_equivariant, f"Spherical harmonics (l_max={l_max}) failed equivariance test"


def test_spherical_harmonics_inversion_equivariance():
    """Test that spherical harmonics have correct parity under inversion."""
    for l in range(6):
        # Generate random positions
        num_positions = 10
        positions = mx.random.normal((num_positions, 3))
        
        # Compute spherical harmonics
        Y = spherical_harmonics(l, positions, normalize=True)
        
        # Compute spherical harmonics at inverted positions
        Y_inverted = spherical_harmonics(l, -positions, normalize=True)
        
        # Expected: Y_l(-r) = (-1)^l * Y_l(r)
        expected_Y = ((-1) ** l) * Y
        
        # Check if they match
        max_error = mx.max(mx.abs(Y_inverted - expected_Y)).item()
        
        print(f"Spherical harmonics l={l} inversion test: max_error = {max_error:.2e}")
        assert max_error < 1e-5, f"Spherical harmonics (l={l}) failed inversion test"


def test_spherical_harmonics_orthogonality():
    """Test that spherical harmonics are orthogonal for different l."""
    num_positions = 100
    positions = mx.random.normal((num_positions, 3))
    
    # Compute spherical harmonics for different l values
    Y1 = spherical_harmonics(1, positions, normalize=True)
    Y2 = spherical_harmonics(2, positions, normalize=True)
    
    # Check orthogonality: integral(Y1 * Y2) should be 0
    # Using discrete approximation
    dot_product = mx.sum(Y1 * Y2, axis=0)
    max_dot = mx.max(mx.abs(dot_product)).item()
    
    print(f"Spherical harmonics orthogonality l=1 vs l=2: max_dot = {max_dot:.2e}")
    assert max_dot < 1e-2, f"Spherical harmonics failed orthogonality test"


def test_spherical_harmonics_normalization():
    """Test that spherical harmonics are properly normalized."""
    num_positions = 1000
    positions = mx.random.normal((num_positions, 3))
    
    # Test normalization for different l values
    for l in [1, 2, 3]:
        Y = spherical_harmonics(l, positions)
        
        # Check normalization: integral(|Y|^2) should be 1
        # Using discrete approximation over unit sphere
        # Normalize positions to unit sphere
        norms = mx.linalg.norm(positions, axis=1, keepdims=True)
        unit_positions = positions / (norms + 1e-8)
        
        Y_unit = spherical_harmonics(l, unit_positions, normalize=True)
        
        # For m components, each should be normalized
        for m_idx in range(2 * l + 1):
            Y_m = Y_unit[:, m_idx]
            norm_squared = mx.mean(Y_m ** 2).item()
            
            # Expected norm should be close to 1/(4pi) for uniform sampling
            expected_norm = 1.0 / (4 * np.pi)
            error = abs(norm_squared - expected_norm)
            
            print(f"Spherical harmonics l={l}, m={m_idx-l}: norm_error = {error:.2e}")
            assert error < 1e-2, f"Spherical harmonics (l={l}, m={m_idx-l}) failed normalization test"


if __name__ == "__main__":
    # Run all tests
    test_spherical_harmonics_rotational_equivariance()
    test_spherical_harmonics_inversion_equivariance()
    test_spherical_harmonics_orthogonality()
    test_spherical_harmonics_normalization()
    print("All spherical harmonics equivariance tests passed!")