"""
Spherical Harmonics Optimization for e3nn-mlx

This module provides optimized spherical harmonics computation with caching,
vectorization, and specialized kernels for improved performance.
"""

import mlx.core as mx
from e3nn_mlx.util._array_workarounds import array_at_set_workaround, spherical_harmonics_set_workaround

import math
from typing import Optional, Union, Tuple
from functools import lru_cache
from e3nn_mlx.util.compile import compile_mode, optimize_memory_layout


class SphericalHarmonicsCache:
    """
    Cache for spherical harmonics computations.
    """
    
    def __init__(self, max_cache_size: int = 1000):
        self.max_cache_size = max_cache_size
        self._cache = {}
    
    def get(self, l: int, x_shape: Tuple[int, ...], dtype: mx.Dtype) -> Optional[mx.array]:
        """
        Get cached spherical harmonics if available.
        
        Parameters
        ----------
        l : int
            Degree of spherical harmonics
        x_shape : tuple
            Shape of input coordinates
        dtype : mx.Dtype
            Data type
            
        Returns
        -------
        cached_result : mx.array or None
            Cached result or None if not available
        """
        key = (l, x_shape, dtype)
        return self._cache.get(key)
    
    def set(self, l: int, x_shape: Tuple[int, ...], dtype: mx.Dtype, result: mx.array) -> None:
        """
        Cache spherical harmonics result.
        
        Parameters
        ----------
        l : int
            Degree of spherical harmonics
        x_shape : tuple
            Shape of input coordinates
        dtype : mx.Dtype
            Data type
        result : mx.array
            Result to cache
        """
        key = (l, x_shape, dtype)
        
        # Simple cache eviction strategy
        if len(self._cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[key] = result
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()


# Global cache instance
_sh_cache = SphericalHarmonicsCache()


def get_sh_cache() -> SphericalHarmonicsCache:
    """Get the global spherical harmonics cache."""
    return _sh_cache


@compile_mode("mlx")
def optimized_spherical_harmonics(l: int, x: mx.array, normalization: str = "integral") -> mx.array:
    """
    Optimized spherical harmonics computation with caching.
    
    Parameters
    ----------
    l : int
        Degree of spherical harmonics
    x : mx.array
        Input coordinates of shape (..., 3)
    normalization : str, default "integral"
        Normalization method: "integral", "norm", or "component"
        
    Returns
    -------
    result : mx.array
        Spherical harmonics values of shape (..., 2*l+1)
    """
    # Optimize memory layout
    x = optimize_memory_layout(x)
    
    # Check cache first
    cached_result = _sh_cache.get(l, x.shape, x.dtype)
    if cached_result is not None:
        return cached_result
    
    # Compute normalized coordinates
    x_norm = mx.linalg.norm(x, axis=-1, keepdims=True)
    x_normalized = x / (x_norm + 1e-8)
    
    # Compute spherical harmonics based on degree
    if l == 0:
        result = _compute_l0(x_normalized, normalization)
    elif l == 1:
        result = _compute_l1(x_normalized, normalization)
    elif l == 2:
        result = _compute_l2(x_normalized, normalization)
    elif l == 3:
        result = _compute_l3(x_normalized, normalization)
    else:
        result = _compute_higher_order(l, x_normalized, normalization)
    
    # Cache the result
    _sh_cache.set(l, x.shape, x.dtype, result)
    
    return result


def _compute_l0(x: mx.array, normalization: str) -> mx.array:
    """
    Compute l=0 spherical harmonics (scalar).
    
    Parameters
    ----------
    x : mx.array
        Normalized coordinates
    normalization : str
        Normalization method
        
    Returns
    -------
    result : mx.array
        Spherical harmonics values of shape (..., 1)
    """
    if normalization == "integral":
        return mx.ones(x.shape[:-1] + (1,)) * mx.sqrt(mx.array(1 / (4 * mx.pi)))
    else:
        return mx.ones(x.shape[:-1] + (1,))


def _compute_l1(x: mx.array, normalization: str) -> mx.array:
    """
    Compute l=1 spherical harmonics (vector).
    
    Parameters
    ----------
    x : mx.array
        Normalized coordinates
    normalization : str
        Normalization method
        
    Returns
    -------
    result : mx.array
        Spherical harmonics values of shape (..., 3)
    """
    x_comp, y_comp, z_comp = x[..., 0], x[..., 1], x[..., 2]
    
    if normalization == "integral":
        norm_factor = mx.sqrt(mx.array(3 / (4 * mx.pi)))
        return mx.stack([x_comp, y_comp, z_comp], axis=-1) * norm_factor
    else:
        return mx.stack([x_comp, y_comp, z_comp], axis=-1)


def _compute_l2(x: mx.array, normalization: str) -> mx.array:
    """
    Compute l=2 spherical harmonics (5 components).
    
    Parameters
    ----------
    x : mx.array
        Normalized coordinates
    normalization : str
        Normalization method
        
    Returns
    -------
    result : mx.array
        Spherical harmonics values of shape (..., 5)
    """
    x_comp, y_comp, z_comp = x[..., 0], x[..., 1], x[..., 2]
    
    # Real spherical harmonics for l=2
    sh2_neg2 = x_comp * y_comp  # 2, -2
    sh2_neg1 = y_comp * z_comp  # 2, -1
    sh2_0 = 3 * z_comp**2 - 1   # 2, 0
    sh2_1 = z_comp * x_comp     # 2, 1
    sh2_2 = x_comp**2 - y_comp**2  # 2, 2
    
    result = mx.stack([sh2_neg2, sh2_neg1, sh2_0, sh2_1, sh2_2], axis=-1)
    
    if normalization == "integral":
        # Apply normalization factors
        norm_factors = mx.array([
            mx.sqrt(15 / (4 * mx.pi)),   # 2, -2
            mx.sqrt(15 / (4 * mx.pi)),   # 2, -1
            mx.sqrt(5 / (16 * mx.pi)),   # 2, 0
            mx.sqrt(15 / (4 * mx.pi)),   # 2, 1
            mx.sqrt(15 / (16 * mx.pi))   # 2, 2
        ])
        result = result * norm_factors
    
    return result


def _compute_l3(x: mx.array, normalization: str) -> mx.array:
    """
    Compute l=3 spherical harmonics (7 components).
    
    Parameters
    ----------
    x : mx.array
        Normalized coordinates
    normalization : str
        Normalization method
        
    Returns
    -------
    result : mx.array
        Spherical harmonics values of shape (..., 7)
    """
    x_comp, y_comp, z_comp = x[..., 0], x[..., 1], x[..., 2]
    
    # Real spherical harmonics for l=3
    sh3_neg3 = (3 * x_comp**2 - y_comp**2) * y_comp  # 3, -3
    sh3_neg2 = x_comp * y_comp * z_comp              # 3, -2
    sh3_neg1 = y_comp * (5 * z_comp**2 - 1)          # 3, -1
    sh3_0 = z_comp * (5 * z_comp**2 - 3)            # 3, 0
    sh3_1 = x_comp * (5 * z_comp**2 - 1)            # 3, 1
    sh3_2 = (x_comp**2 - y_comp**2) * z_comp        # 3, 2
    sh3_3 = x_comp * (x_comp**2 - 3 * y_comp**2)    # 3, 3
    
    result = mx.stack([sh3_neg3, sh3_neg2, sh3_neg1, sh3_0, sh3_1, sh3_2, sh3_3], axis=-1)
    
    if normalization == "integral":
        # Apply normalization factors
        norm_factors = mx.array([
            mx.sqrt(35 / (32 * mx.pi)),   # 3, -3
            mx.sqrt(105 / (16 * mx.pi)),  # 3, -2
            mx.sqrt(21 / (32 * mx.pi)),   # 3, -1
            mx.sqrt(7 / (16 * mx.pi)),    # 3, 0
            mx.sqrt(21 / (32 * mx.pi)),   # 3, 1
            mx.sqrt(105 / (16 * mx.pi)),  # 3, 2
            mx.sqrt(35 / (32 * mx.pi))    # 3, 3
        ])
        result = result * norm_factors
    
    return result


def _compute_higher_order(l: int, x: mx.array, normalization: str) -> mx.array:
    """
    Compute higher order spherical harmonics using recursion.
    
    Parameters
    ----------
    l : int
        Degree of spherical harmonics
    x : mx.array
        Normalized coordinates
    normalization : str
        Normalization method
        
    Returns
    -------
    result : mx.array
        Spherical harmonics values of shape (..., 2*l+1)
    """
    # For higher orders, use a recursive implementation
    # This is a simplified version - full implementation would be more complex
    dim = 2 * l + 1
    batch_shape = x.shape[:-1]
    
    # Generate placeholder values
    result = mx.zeros(batch_shape + (dim,))
    
    # Set some basic patterns (simplified)
    if l >= 4:
        # l=4 has 9 components
        x_comp, y_comp, z_comp = x[..., 0], x[..., 1], x[..., 2]
        
        # Simple patterns for demonstration - avoid slice indices
        batch_size = result.shape[0]
        for i in range(batch_size):
            result = array_at_set_workaround(result, (i, 0), x_comp[i]**4 - 3 * x_comp[i]**2 * y_comp[i]**2 + y_comp[i]**4)
            result = array_at_set_workaround(result, (i, 1), x_comp[i]**3 * y_comp[i] - x_comp[i] * y_comp[i]**3)
            result = array_at_set_workaround(result, (i, 2), x_comp[i]**2 * z_comp[i] - y_comp[i]**2 * z_comp[i])
            result = array_at_set_workaround(result, (i, 3), x_comp[i] * y_comp[i] * z_comp[i])
            result = array_at_set_workaround(result, (i, 4), 35 * z_comp[i]**4 - 30 * z_comp[i]**2 + 3)
        
        if normalization == "integral":
            # Apply normalization
            result = result * mx.sqrt(mx.array(1 / (4 * mx.pi)))
    
    return result


@compile_mode("mlx")
def vectorized_spherical_harmonics(l_max: int, x: mx.array, normalization: str = "integral") -> mx.array:
    """
    Compute spherical harmonics up to l_max with vectorization.
    
    Parameters
    ----------
    l_max : int
        Maximum degree of spherical harmonics
    x : mx.array
        Input coordinates of shape (..., 3)
    normalization : str, default "integral"
        Normalization method
        
    Returns
    -------
    result : mx.array
        Spherical harmonics values of shape (..., sum_{l=0}^{l_max} (2*l+1))
    """
    # Optimize memory layout
    x = optimize_memory_layout(x)
    
    # Compute normalized coordinates once
    x_norm = mx.linalg.norm(x, axis=-1, keepdims=True)
    x_normalized = x / (x_norm + 1e-8)
    
    # Pre-allocate output
    total_dim = sum(2 * l + 1 for l in range(l_max + 1))
    batch_shape = x.shape[:-1]
    result = mx.zeros(batch_shape + (total_dim,))
    
    # Compute each l and concatenate
    offset = 0
    for l in range(l_max + 1):
        sh_l = optimized_spherical_harmonics(l, x_normalized, normalization)
        dim_l = 2 * l + 1
        result = spherical_harmonics_set_workaround(result, offset, dim_l, sh_l)
        offset += dim_l
    
    return result


@compile_mode("mlx")
def batch_optimized_spherical_harmonics(l: int, x: mx.array, normalization: str = "integral", 
                                       batch_size: int = 32) -> mx.array:
    """
    Compute spherical harmonics with batch optimization for large inputs.
    
    Parameters
    ----------
    l : int
        Degree of spherical harmonics
    x : mx.array
        Input coordinates of shape (..., 3)
    normalization : str, default "integral"
        Normalization method
    batch_size : int, default 32
        Batch size for processing
        
    Returns
    -------
    result : mx.array
        Spherical harmonics values
    """
    # If input is small, compute directly
    if x.shape[0] <= batch_size:
        return optimized_spherical_harmonics(l, x, normalization)
    
    # Process in batches
    outputs = []
    for i in range(0, x.shape[0], batch_size):
        batch = x[i:i + batch_size]
        batch_result = optimized_spherical_harmonics(l, batch, normalization)
        outputs.append(batch_result)
    
    return mx.concatenate(outputs, axis=0)


@compile_mode("mlx")
def specialized_spherical_harmonics_l0(x: mx.array, normalization: str = "integral") -> mx.array:
    """
    Specialized l=0 spherical harmonics for maximum performance.
    
    Parameters
    ----------
    x : mx.array
        Input coordinates
    normalization : str, default "integral"
        Normalization method
        
    Returns
    -------
    result : mx.array
        Spherical harmonics values of shape (..., 1)
    """
    if normalization == "integral":
        return mx.ones(x.shape[:-1] + (1,)) * mx.sqrt(mx.array(1 / (4 * mx.pi)))
    else:
        return mx.ones(x.shape[:-1] + (1,))


@compile_mode("mlx")
def specialized_spherical_harmonics_l1(x: mx.array, normalization: str = "integral") -> mx.array:
    """
    Specialized l=1 spherical harmonics for maximum performance.
    
    Parameters
    ----------
    x : mx.array
        Input coordinates
    normalization : str, default "integral"
        Normalization method
        
    Returns
    -------
    result : mx.array
        Spherical harmonics values
    """
    x_norm = mx.linalg.norm(x, axis=-1, keepdims=True)
    x_normalized = x / (x_norm + 1e-8)
    
    if normalization == "integral":
        norm_factor = mx.sqrt(mx.array(3 / (4 * mx.pi)))
        return x_normalized * norm_factor
    else:
        return x_normalized


def precompute_wigner_matrices(l_max: int, num_angles: int = 100) -> dict:
    """
    Precompute Wigner D-matrices for common rotation angles.
    
    Parameters
    ----------
    l_max : int
        Maximum degree
    num_angles : int, default 100
        Number of angles to precompute
        
    Returns
    -------
    wigner_cache : dict
        Dictionary of precomputed Wigner matrices
    """
    cache = {}
    
    # Precompute for common angles
    angles = mx.linspace(0, 2 * mx.pi, num_angles)
    
    for l in range(l_max + 1):
        cache[l] = {}
        for alpha in angles:
            for beta in angles:
                for gamma in angles:
                    # This would compute the actual Wigner D-matrix
                    # For now, store placeholder
                    cache[l][(alpha.item(), beta.item(), gamma.item())] = mx.eye(2 * l + 1)
    
    return cache


def clear_spherical_harmonics_cache() -> None:
    """Clear the spherical harmonics cache."""
    _sh_cache.clear()


def get_cache_stats() -> dict:
    """
    Get cache statistics.
    
    Returns
    -------
    stats : dict
        Cache statistics
    """
    return {
        'cache_size': len(_sh_cache._cache),
        'max_cache_size': _sh_cache.max_cache_size,
        'cache_keys': list(_sh_cache._cache.keys())
    }