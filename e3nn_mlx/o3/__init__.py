"""
e3nn.o3: O(3) equivariant operations

This module provides the core operations for working with O(3) equivariant neural networks,
including tensor products, spherical harmonics, rotations, and linear transformations.
"""

# Import the irrep module to enable clean Irrep lookup
from . import irrep

from ._irreps import Irrep, Irreps
from ._spherical_harmonics import spherical_harmonics, spherical_harmonics_alpha, spherical_harmonics_beta
from ._wigner import wigner_D, wigner_3j, change_basis_real_to_complex, su2_generators, so3_generators, clebsch_gordan
from ._rotation import (
    identity, rand, rand_matrix, axis_angle, angles_to_matrix,
    matrix_to_angles, angles_to_xyz, xyz_to_angles, angles_to_axis_angle, angles_to_quaternion,
    matrix_to_axis_angle, matrix_to_quaternion, axis_angle_to_angles, axis_angle_to_matrix,
    axis_angle_to_quaternion, quaternion_to_angles, quaternion_to_matrix, quaternion_to_axis_angle
)
from ._tensor_product import TensorProduct, FullyConnectedTensorProduct, ElementwiseTensorProduct, FullTensorProduct, TensorSquare
from ._linear import Linear
from ._norm import Norm
from ._reduce import ReducedTensorProducts
from ._s2grid import ToS2Grid, FromS2Grid, s2_grid, spherical_harmonics_s2_grid
from ._so3grid import SO3Grid
from ._angular_spherical_harmonics import spherical_harmonics_alpha_beta
from ._legendre import Legendre, legendre_polynomial
from ._fft import rfft, irfft, fft, ifft, fftshift, ifftshift

__all__ = [
    # Core classes
    "Irrep",
    "Irreps",
    "irrep",
    "TensorProduct",
    "FullyConnectedTensorProduct",
    "ElementwiseTensorProduct", 
    "FullTensorProduct",
    "TensorSquare",
    "Linear",
    "Norm",
    "ReducedTensorProducts",
    "ToS2Grid",
    "FromS2Grid",
    "SO3Grid",
    
    # Spherical harmonics
    "spherical_harmonics",
    "spherical_harmonics_alpha",
    "spherical_harmonics_beta",
    "spherical_harmonics_alpha_beta",
    
    # Wigner matrices
    "wigner_D",
    "wigner_3j",
    "change_basis_real_to_complex",
    "su2_generators",
    "so3_generators",
    "clebsch_gordan",
    
    # Rotation operations
    "identity",
    "rand",
    "rand_matrix",
    "axis_angle",
    "angles_to_matrix",
    "matrix_to_angles",
    "angles_to_xyz",
    "xyz_to_angles",
    "angles_to_axis_angle",
    "angles_to_quaternion",
    "matrix_to_axis_angle",
    "matrix_to_quaternion",
    "axis_angle_to_angles",
    "axis_angle_to_matrix",
    "axis_angle_to_quaternion",
    "quaternion_to_angles",
    "quaternion_to_matrix",
    "quaternion_to_axis_angle",
    
    # Grid utilities
    "s2_grid",
    "spherical_harmonics_s2_grid",
    
    # Legendre polynomials
    "Legendre",
    "legendre_polynomial",
    
    # FFT functions
    "rfft",
    "irfft", 
    "fft",
    "ifft",
    "fftshift",
    "ifftshift",
]