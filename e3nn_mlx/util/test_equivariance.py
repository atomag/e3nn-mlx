"""
Equivariance validation utilities for e3nn-mlx.

This module provides comprehensive tools to test whether operations are equivariant
under rotations and other symmetry transformations.
"""

import mlx.core as mx
import numpy as np
from typing import Callable, List, Tuple, Optional, Union
from ..o3 import Irreps, wigner_D, angles_to_matrix, rand_matrix, matrix_to_angles


class EquivarianceTester:
    """
    Comprehensive equivariance testing framework for e3nn-mlx operations.
    
    This class provides methods to test whether operations respect the equivariance
    properties required for E(3)-equivariant neural networks.
    """
    
    def __init__(self, tolerance: float = 1e-5, num_samples: int = 10):
        """
        Initialize the equivariance tester.
        
        Parameters
        ----------
        tolerance : float
            Maximum allowed error for equivariance to hold
        num_samples : int
            Number of random test samples to generate
        """
        self.tolerance = tolerance
        self.num_samples = num_samples
        
    def assert_equivariant(
        self,
        operation: Callable,
        irreps_in: Union[str, Irreps],
        irreps_out: Union[str, Irreps],
        operation_kwargs: Optional[dict] = None,
        input_generator: Optional[Callable] = None,
        test_rotations: bool = True,
        test_inversions: bool = True,
        test_translations: bool = False
    ) -> Tuple[bool, dict]:
        """
        Test if an operation is equivariant under various transformations.
        
        Parameters
        ----------
        operation : Callable
            The operation to test
        irreps_in : str or Irreps
            Input irreducible representations
        irreps_out : str or Irreps
            Output irreducible representations
        operation_kwargs : dict, optional
            Additional keyword arguments for the operation
        input_generator : Callable, optional
            Function to generate test inputs
        test_rotations : bool
            Whether to test rotational equivariance
        test_inversions : bool
            Whether to test inversion equivariance
        test_translations : bool
            Whether to test translational equivariance
            
        Returns
        -------
        Tuple[bool, dict]
            (is_equivariant, results_dict)
        """
        if operation_kwargs is None:
            operation_kwargs = {}
            
        irreps_in = Irreps(irreps_in)
        irreps_out = Irreps(irreps_out)
        
        results = {
            'rotations': {'passed': True, 'max_error': 0.0, 'errors': []},
            'inversions': {'passed': True, 'max_error': 0.0, 'errors': []},
            'translations': {'passed': True, 'max_error': 0.0, 'errors': []}
        }
        
        # Test rotational equivariance
        if test_rotations:
            rot_passed, rot_errors = self._test_rotational_equivariance(
                operation, irreps_in, irreps_out, operation_kwargs, input_generator
            )
            results['rotations']['passed'] = rot_passed
            results['rotations']['errors'] = rot_errors
            results['rotations']['max_error'] = max(rot_errors) if rot_errors else 0.0
            
        # Test inversion equivariance
        if test_inversions:
            inv_passed, inv_errors = self._test_inversion_equivariance(
                operation, irreps_in, irreps_out, operation_kwargs, input_generator
            )
            results['inversions']['passed'] = inv_passed
            results['inversions']['errors'] = inv_errors
            results['inversions']['max_error'] = max(inv_errors) if inv_errors else 0.0
            
        # Test translational equivariance
        if test_translations:
            trans_passed, trans_errors = self._test_translational_equivariance(
                operation, irreps_in, irreps_out, operation_kwargs, input_generator
            )
            results['translations']['passed'] = trans_passed
            results['translations']['errors'] = trans_errors
            results['translations']['max_error'] = max(trans_errors) if trans_errors else 0.0
            
        # Overall result
        is_equivariant = all([
            results['rotations']['passed'] if test_rotations else True,
            results['inversions']['passed'] if test_inversions else True,
            results['translations']['passed'] if test_translations else True
        ])
        
        return is_equivariant, results
        
    def _test_rotational_equivariance(
        self,
        operation: Callable,
        irreps_in: Irreps,
        irreps_out: Irreps,
        operation_kwargs: dict,
        input_generator: Optional[Callable]
    ) -> Tuple[bool, List[float]]:
        """Test rotational equivariance."""
        errors = []
        
        for i in range(self.num_samples):
            # Generate random rotation matrix
            R = rand_matrix()
            
            # Generate input or use provided generator
            if input_generator is None:
                x = self._generate_random_input(irreps_in)
            else:
                x = input_generator(irreps_in)
                
            # Apply operation first, then rotate
            op_x = operation(x, **operation_kwargs)
            rotated_op_x = self._apply_rotation(op_x, irreps_out, R)
            
            # Rotate first, then apply operation
            rotated_x = self._apply_rotation(x, irreps_in, R)
            op_rotated_x = operation(rotated_x, **operation_kwargs)
            
            # Compute error
            error = mx.max(mx.abs(rotated_op_x - op_rotated_x)).item()
            errors.append(error)
            
        max_error = max(errors)
        passed = max_error < self.tolerance
        
        return passed, errors
        
    def _test_inversion_equivariance(
        self,
        operation: Callable,
        irreps_in: Irreps,
        irreps_out: Irreps,
        operation_kwargs: dict,
        input_generator: Optional[Callable]
    ) -> Tuple[bool, List[float]]:
        """Test inversion equivariance."""
        errors = []
        
        for i in range(self.num_samples):
            # Generate input or use provided generator
            if input_generator is None:
                x = self._generate_random_input(irreps_in)
            else:
                x = input_generator(irreps_in)
                
            # Apply operation first, then invert
            op_x = operation(x, **operation_kwargs)
            inverted_op_x = self._apply_inversion(op_x, irreps_out)
            
            # Invert first, then apply operation
            inverted_x = self._apply_inversion(x, irreps_in)
            op_inverted_x = operation(inverted_x, **operation_kwargs)
            
            # Compute error
            error = mx.max(mx.abs(inverted_op_x - op_inverted_x)).item()
            errors.append(error)
            
        max_error = max(errors)
        passed = max_error < self.tolerance
        
        return passed, errors
        
    def _test_translational_equivariance(
        self,
        operation: Callable,
        irreps_in: Irreps,
        irreps_out: Irreps,
        operation_kwargs: dict,
        input_generator: Optional[Callable]
    ) -> Tuple[bool, List[float]]:
        """Test translational equivariance (for position-dependent operations)."""
        errors = []
        
        for i in range(self.num_samples):
            # Generate random translation
            t = mx.random.normal((3,))
            
            # Generate input with positions or use provided generator
            if input_generator is None:
                x, positions = self._generate_random_input_with_positions(irreps_in)
            else:
                x, positions = input_generator(irreps_in)
                
            # Apply operation first, then translate
            op_x = operation(x, positions, **operation_kwargs)
            translated_op_x = self._apply_translation(op_x, positions, irreps_out, t)
            
            # Translate first, then apply operation
            translated_positions = positions + t
            translated_x = x  # Features don't change under translation
            op_translated_x = operation(translated_x, translated_positions, **operation_kwargs)
            
            # Compute error
            error = mx.max(mx.abs(translated_op_x - op_translated_x)).item()
            errors.append(error)
            
        max_error = max(errors)
        passed = max_error < self.tolerance
        
        return passed, errors
        
    def _generate_random_input(self, irreps: Irreps) -> mx.array:
        """Generate random input features for given irreps."""
        features = []
        
        for mul, (l, p) in irreps:
            dim = mul * (2 * l + 1)
            features.append(mx.random.normal((dim,)))
            
        return mx.concatenate(features)
        
    def _generate_random_input_with_positions(self, irreps: Irreps) -> Tuple[mx.array, mx.array]:
        """Generate random input features with positions."""
        # Generate features
        features = self._generate_random_input(irreps)
        
        # Generate random positions
        num_particles = 10  # Fixed number of particles for simplicity
        positions = mx.random.normal((num_particles, 3))
        
        # Repeat features for each particle
        features = mx.tile(features, (num_particles,))
        
        return features, positions
        
    def _apply_rotation(self, x: mx.array, irreps: Irreps, R: mx.array) -> mx.array:
        """Apply rotation to features."""
        output = []
        index = 0

        for mul, (l, p) in irreps:
            dim = mul * (2 * l + 1)
            chunk = x[index:index + dim]
            index += dim
            
            if l == 0:
                # Scalars are invariant under rotation
                output.append(chunk)
            else:
                # Apply Wigner D-matrix for each l
                # Convert rotation matrix to Euler angles (Y-X-Y convention)
                a, b, c = matrix_to_angles(R)
                D = wigner_D(l, a, b, c)
                
                # Reshape and apply rotation
                chunk_reshaped = chunk.reshape(mul, 2 * l + 1)
                rotated_chunk = chunk_reshaped @ D.T
                rotated_chunk = rotated_chunk.reshape(dim)
                
                output.append(rotated_chunk)
                
        return mx.concatenate(output)
        
    def _apply_inversion(self, x: mx.array, irreps: Irreps) -> mx.array:
        """Apply inversion (parity transformation) to features."""
        output = []
        index = 0
        
        for mul, (l, p) in irreps:
            dim = mul * (2 * l + 1)
            chunk = x[index:index + dim]
            index += dim
            
            # Apply O(3) parity: multiply by irrep parity p (±1)
            output.append(p * chunk)
            
        return mx.concatenate(output)
        
    def _apply_translation(self, x: mx.array, positions: mx.array, irreps: Irreps, t: mx.array) -> mx.array:
        """Apply translation to features (for position-dependent operations)."""
        # For most operations, translation doesn't affect the features directly
        # This is a placeholder for more complex translation handling
        return x


def assert_equivariant(
    operation: Callable,
    irreps_in: Union[str, Irreps],
    irreps_out: Union[str, Irreps],
    operation_kwargs: Optional[dict] = None,
    tolerance: float = 1e-5,
    num_samples: int = 10
) -> bool:
    """
    Convenience function to test equivariance of an operation.
    
    Parameters
    ----------
    operation : Callable
        The operation to test
    irreps_in : str or Irreps
        Input irreducible representations
    irreps_out : str or Irreps
        Output irreducible representations
    operation_kwargs : dict, optional
        Additional keyword arguments for the operation
    tolerance : float
        Maximum allowed error for equivariance to hold
    num_samples : int
        Number of random test samples to generate
        
    Returns
    -------
    bool
        True if the operation is equivariant, False otherwise
    """
    tester = EquivarianceTester(tolerance=tolerance, num_samples=num_samples)
    is_equivariant, results = tester.assert_equivariant(
        operation, irreps_in, irreps_out, operation_kwargs
    )
    
    if not is_equivariant:
        print(f"Equivariance test failed!")
        for transformation, result in results.items():
            if not result['passed']:
                print(f"  {transformation}: max_error = {result['max_error']:.2e}")
    else:
        print(f"Equivariance test passed with max_error < {tolerance:.1e}")
        
    return is_equivariant
