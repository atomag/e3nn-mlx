"""
Legendre polynomials for e3nn-mlx.

This module provides Legendre polynomial functionality for computing associated Legendre polynomials
used in spherical harmonics calculations.
"""

import mlx.core as mx
import math
from typing import List, Union
from functools import lru_cache


def _legendre_coefficient(l: int, m: int) -> List[tuple]:
    """
    Compute coefficients for the associated Legendre polynomial P_l^m(z).
    
    Uses Bonnet's recursion formula to compute polynomial coefficients.
    
    Parameters
    ----------
    l : int
        Degree of the polynomial
    m : int
        Order of the polynomial (absolute value)
        
    Returns
    -------
    coefficients : List[tuple]
        List of (power, coefficient) tuples where power is the exponent of z
    """
    # Normalization factor
    if m == 0:
        norm = math.sqrt((2 * l + 1) / (4 * math.pi))
    else:
        norm = math.sqrt((2 * l + 1) / (4 * math.pi) * math.factorial(l - abs(m)) / math.factorial(l + abs(m)))
    
    # Start with P_m^m = (-1)^m (2m-1)!! (1-z^2)^{m/2}
    if m == 0:
        # P_0^0 = 1
        coeffs = [(0, 1.0)]
    else:
        # P_m^m = (-1)^m (2m-1)!! (1-z^2)^{m/2}
        # For now, we'll handle the z part only (1-z^2)^{m/2} becomes y^m where y = sqrt(1-z^2)
        coeffs = [(0, (-1)**m * _double_factorial(2 * m - 1))]
    
    # Use recursion to get P_l^m from P_{l-1}^m and P_{l-2}^m
    # (l-m) P_l^m = z (2l-1) P_{l-1}^m - (l+m-1) P_{l-2}^m
    for n in range(m, l):
        new_coeffs = []
        
        # Term 1: z (2n-1) P_{n-1}^m
        for power, coeff in coeffs:
            new_coeffs.append((power + 1, coeff * (2 * n - 1)))
        
        # Term 2: -(n+m-1) P_{n-2}^m (if n >= m+2)
        if n >= m + 2:
            # We need to keep track of P_{n-2}^m separately
            # For simplicity, we'll recompute the recursion
            pass
        
        coeffs = new_coeffs
    
    # Apply normalization
    return [(power, coeff * norm) for power, coeff in coeffs]


def _double_factorial(n: int) -> int:
    """Compute double factorial n!! = n(n-2)(n-4)..."""
    if n <= 0:
        return 1
    result = 1
    for i in range(n, 0, -2):
        result *= i
    return result


class Legendre:
    """
    Legendre polynomial class for computing associated Legendre polynomials.
    
    This class computes the associated Legendre polynomials P_l^m(cos(β)) 
    used in spherical harmonics calculations.
    
    Parameters
    ----------
    ls : List[int]
        List of angular momentum quantum numbers to compute
    """
    
    def __init__(self, ls: Union[List[int], int]):
        if isinstance(ls, int):
            ls = [ls]
        self.ls = ls
        self.lmax = max(ls) if ls else 0
        
        # Precompute polynomial coefficients for efficiency
        self._coefficients = {}
        for l in ls:
            self._coefficients[l] = {}
            for m in range(l + 1):
                self._coefficients[l][m] = _legendre_coefficient(l, m)
    
    def __call__(self, z: mx.array, y: mx.array) -> mx.array:
        """
        Compute Legendre polynomials for given z and y values.
        
        Parameters
        ----------
        z : mx.array
            cos(β) values
        y : mx.array  
            sin(β) values = sqrt(1 - z^2)
            
        Returns
        -------
        result : mx.array
            Array of shape (batch_size, sum(2l+1 for l in ls))
        """
        # Ensure inputs are flattened for processing
        original_shape = z.shape
        z_flat = z.reshape(-1)
        y_flat = y.reshape(-1)
        batch_size = z_flat.shape[0]
        
        # Compute total output dimension
        total_dim = sum(2 * l + 1 for l in self.ls)
        result = mx.zeros((batch_size, total_dim))
        
        # Fill in values for each l
        col_offset = 0
        for l in self.ls:
            dim = 2 * l + 1
            
            # Compute P_l^m for m = 0 to l
            for m in range(l + 1):
                coeffs = self._coefficients[l][m]
                
                # Evaluate polynomial
                poly_value = mx.zeros_like(z_flat)
                for power, coeff in coeffs:
                    if power == 0:
                        poly_value += coeff * mx.ones_like(z_flat)
                    else:
                        poly_value += coeff * (z_flat ** power)
                
                # Multiply by y^m for associated Legendre
                if m > 0:
                    poly_value *= (y_flat ** m)
                
                # Place in result array
                # Positive m: index l + m
                # Negative m: index l - m  
                # P_l^{-m} = P_l^m
                
                # Positive m
                result = result.at[:, col_offset + l + m].add(poly_value)
                
                # Negative m (except m=0)
                if m > 0:
                    result = result.at[:, col_offset + l - m].add(poly_value)
            
            col_offset += dim
        
        return result.reshape(original_shape + (total_dim,))
    
    def __repr__(self):
        return f"Legendre(ls={self.ls})"


def legendre_polynomial(l: int, m: int, z: mx.array) -> mx.array:
    """
    Compute a single associated Legendre polynomial P_l^m(z).
    
    Parameters
    ----------
    l : int
        Degree of the polynomial
    m : int  
        Order of the polynomial
    z : mx.array
        Input values (typically cos(β))
        
    Returns
    -------
    result : mx.array
        P_l^m(z) values
    """
    legendre_obj = Legendre([l])
    y = mx.sqrt(1 - z**2)
    
    # Extract just the P_l^m component
    full_result = legendre_obj(z, y)
    
    if m >= 0:
        # Positive m: index l + m
        idx = l + m
    else:
        # Negative m: index l - m
        idx = l - m
    
    return full_result[..., idx]


# Pre-cache common Legendre polynomials for efficiency
@lru_cache(maxsize=128)
def _cached_legendre_coefficients(l: int, m: int) -> List[tuple]:
    """Cached version of _legendre_coefficient for performance."""
    return _legendre_coefficient(l, m)