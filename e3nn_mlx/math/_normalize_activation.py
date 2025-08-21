"""
Normalization utilities for activation functions.
"""

import mlx.core as mx
import math
from typing import Callable, Union


def normalize2mom(x: Union[Callable, mx.array], dim: int = -1, eps: float = 1e-5, dtype: mx.Dtype = mx.float32):
    """Normalize to unit second moment.

    - If ``x`` is a callable, returns a wrapper that rescales its output so that E[f(z)^2] = 1 for z ~ N(0,1).
    - If ``x`` is an array, normalize it to zero mean and unit variance along ``dim``.
    """
    if callable(x):
        return Normalize2Mom(x, dtype=dtype)
    # Treat as array normalization (compat with some call sites)
    mean = mx.mean(x, axis=dim, keepdims=True)
    var = mx.var(x, axis=dim, keepdims=True)
    return (x - mean) / mx.sqrt(var + eps)


def moment(f: Callable, n: int, dtype: mx.Dtype = mx.float32) -> float:
    """
    Compute the nth moment <f(z)^n> for z following a normal distribution.
    
    Parameters
    ----------
    f : Callable
        Function to compute moments for
    n : int
        Order of the moment to compute
    dtype : mx.Dtype, default=mx.float32
        Data type for computation
        
    Returns
    -------
    moment_value : float
        The nth moment of f(z) where z ~ N(0, 1)
    """
    # Generate random samples from normal distribution
    # Use a fixed seed for reproducibility
    mx.random.seed(0)
    z = mx.random.normal((100_000,), dtype=dtype)  # Reduced size for speed
    
    # Compute f(z) and then the nth moment
    fz = f(z)
    if n == 2:
        # For second moment, use mean of squares
        return mx.mean(fz * fz).item()
    else:
        # For general nth moment
        return mx.mean(mx.power(fz, n)).item()


class Normalize2Mom:
    """
    Normalize a function to have unit second moment.
    
    This class creates a wrapper around a function that normalizes its output
    such that the second moment equals 1.
    """
    
    def __init__(self, f: Callable, dtype: mx.Dtype = mx.float32):
        """
        Initialize the normalizer.
        
        Parameters
        ----------
        f : Callable
            Function to normalize
        dtype : mx.Dtype, default=mx.float32
            Data type for computation
        """
        self.f = f
        
        # Compute normalization constant
        mx.random.seed(0)  # Ensure reproducibility
        cst = moment(f, 2, dtype=dtype)
        cst = cst ** (-0.5)
        
        # Check if normalization is needed
        self._is_id = abs(cst - 1.0) < 1e-4
        self.cst = cst
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply the normalized function.
        
        Parameters
        ----------
        x : mx.array
            Input tensor
            
        Returns
        -------
        result : mx.array
            Normalized output
        """
        if self._is_id:
            return self.f(x)
        else:
            return self.f(x) * self.cst
    
    def __repr__(self):
        return f"Normalize2Mom(f={self.f}, cst={self.cst:.6f})"


# Backward compatibility alias
normalize2mom_class = Normalize2Mom
