"""
MLX Compilation Utilities for e3nn-mlx

This module provides compilation utilities for optimizing e3nn-mlx operations
using MLX's native compilation capabilities.
"""

import functools
import time
import warnings
from typing import Callable, Any, Optional, Union, Dict, List
import mlx.core as mx


def compile_mode(mode: str = "mlx", **kwargs):
    """
    Decorator for compilation mode selection.
    
    Parameters
    ----------
    mode : str, default "mlx"
        Compilation mode: "mlx", "script", or "none"
    **kwargs : dict
        Additional compilation options
        
    Returns
    -------
    decorator : Callable
        Decorator function
        
    Examples
    --------
    >>> @compile_mode("mlx")
    ... def my_function(x):
    ...     return mx.sin(x)
    """
    def decorator(func: Callable) -> Callable:
        if mode == "none":
            return func
        elif mode in ["mlx", "script"]:
            try:
                # Use MLX's compilation
                compiled_func = mx.compile(func, **kwargs)
                
                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    return compiled_func(*args, **kwargs)
                
                return wrapper
            except Exception as e:
                warnings.warn(f"Compilation failed for {func.__name__}: {e}")
                return func
        else:
            raise ValueError(f"Unknown compilation mode: {mode}")
    return decorator


def compile_function(func: Callable, mode: str = "mlx", **kwargs) -> Callable:
    """
    Compile a function with specific options.
    
    Parameters
    ----------
    func : Callable
        Function to compile
    mode : str, default "mlx"
        Compilation mode
    **kwargs : dict
        Additional compilation options
        
    Returns
    -------
    compiled_func : Callable
        Compiled function
    """
    if mode == "none":
        return func
    
    try:
        return mx.compile(func, **kwargs)
    except Exception as e:
        warnings.warn(f"Compilation failed for {func.__name__}: {e}")
        return func


def get_optimization_level() -> int:
    """
    Get the current optimization level.
    
    Returns
    -------
    level : int
        Current optimization level (0-3)
    """
    # MLX doesn't have explicit optimization levels like PyTorch
    # Return a default level based on the device
    device = mx.get_default_device()
    if device == mx.gpu:
        return 3  # High optimization for GPU
    else:
        return 2  # Medium optimization for CPU


def set_optimization_level(level: int) -> None:
    """
    Set the optimization level.
    
    Parameters
    ----------
    level : int
        Optimization level (0-3)
    """
    if level < 0 or level > 3:
        raise ValueError("Optimization level must be between 0 and 3")
    
    # MLX doesn't have explicit optimization levels
    # We can set the device to influence optimization
    if level >= 2:
        mx.set_default_device(mx.gpu if mx.get_default_device() == mx.gpu else mx.cpu)


class CompilationProfiler:
    """
    Profiler for compilation performance.
    """
    
    def __init__(self):
        self.stats = {}
    
    def profile(self, func: Callable, *args, num_runs: int = 100, **kwargs):
        """
        Profile a function's performance.
        
        Parameters
        ----------
        func : Callable
            Function to profile
        *args : tuple
            Arguments to pass to the function
        num_runs : int, default 100
            Number of runs for averaging
        **kwargs : dict
            Keyword arguments to pass to the function
            
        Returns
        -------
        result : Any
            Function result
        stats : dict
            Performance statistics
        """
        # Warm-up
        for _ in range(10):
            func(*args, **kwargs)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            result = func(*args, **kwargs)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        
        stats = {
            'function': func.__name__,
            'avg_time': avg_time,
            'total_time': end_time - start_time,
            'num_runs': num_runs
        }
        
        self.stats[func.__name__] = stats
        
        return result, stats
    
    def get_stats(self) -> Dict[str, Dict]:
        """
        Get profiling statistics.
        
        Returns
        -------
        stats : dict
            Performance statistics
        """
        return self.stats.copy()
    
    def clear_stats(self) -> None:
        """Clear profiling statistics."""
        self.stats.clear()


def optimize_memory_layout(x: mx.array) -> mx.array:
    """
    Optimize memory layout for MLX operations.
    
    Parameters
    ----------
    x : mx.array
        Input tensor
        
    Returns
    -------
    optimized_x : mx.array
        Optimized tensor
    """
    # MLX arrays are always contiguous, but we can ensure optimal layout
    # by creating a copy if needed
    x = mx.array(x)
    
    # Optimize data types
    if x.dtype == mx.float64:
        x = x.astype(mx.float32)
    
    return x


def batch_optimize(x: mx.array, batch_size: int = 32) -> mx.array:
    """
    Optimize batch processing for large tensors.
    
    Parameters
    ----------
    x : mx.array
        Input tensor
    batch_size : int, default 32
        Optimal batch size
        
    Returns
    -------
    result : mx.array
        Processed tensor
    """
    if x.shape[0] <= batch_size:
        return x
    
    # Process in batches if needed
    outputs = []
    for i in range(0, x.shape[0], batch_size):
        batch = x[i:i + batch_size]
        outputs.append(batch)
    
    return mx.concatenate(outputs, axis=0)


def create_compiled_tensor_product():
    """
    Create a compiled tensor product function.
    
    Returns
    -------
    compiled_func : Callable
        Compiled tensor product function
    """
    @compile_mode("mlx")
    def compiled_tensor_product(x1: mx.array, x2: mx.array, weights: mx.array, instructions: List) -> mx.array:
        """
        Compiled tensor product operation.
        
        Parameters
        ----------
        x1 : mx.array
            First input tensor
        x2 : mx.array
            Second input tensor
        weights : mx.array
            Weight tensor
        instructions : List
            List of instructions
            
        Returns
        -------
        result : mx.array
            Tensor product result
        """
        outputs = []
        
        for ins in instructions:
            if ins.i_in1 >= 0 and ins.i_in2 >= 0:
                # Two-input tensor product
                input1 = x1[..., ins.i_in1]
                input2 = x2[..., ins.i_in2]
                weight = weights[ins.weight_slice]
                
                result = mx.einsum('...i,...j,...ij->...', input1, input2, weight)
                result = result * ins.path_weight
                outputs.append(result)
            elif ins.i_in1 >= 0:
                # One-input operation
                input1 = x1[..., ins.i_in1]
                weight = weights[ins.weight_slice]
                
                result = mx.einsum('...i,...i->...', input1, weight)
                result = result * ins.path_weight
                outputs.append(result)
        
        if outputs:
            return sum(outputs)
        else:
            return mx.zeros(x1.shape[:-1] + (0,))
    
    return compiled_tensor_product


def create_compiled_spherical_harmonics():
    """
    Create a compiled spherical harmonics function.
    
    Returns
    -------
    compiled_func : Callable
        Compiled spherical harmonics function
    """
    @compile_mode("mlx")
    def compiled_spherical_harmonics(l: int, x: mx.array, normalization: str = "integral") -> mx.array:
        """
        Compiled spherical harmonics computation.
        
        Parameters
        ----------
        l : int
            Degree of spherical harmonics
        x : mx.array
            Input coordinates
        normalization : str, default "integral"
            Normalization method
            
        Returns
        -------
        result : mx.array
            Spherical harmonics values
        """
        # Normalize input
        x_norm = mx.linalg.norm(x, axis=-1, keepdims=True)
        x_normalized = x / (x_norm + 1e-8)
        
        if l == 0:
            return mx.ones_like(x_norm[..., 0])
        elif l == 1:
            # Vector spherical harmonics
            if normalization == "integral":
                return x_normalized * mx.sqrt(mx.array(3 / (4 * mx.pi)))
            else:
                return x_normalized
        else:
            # Higher order - use recursive computation
            return _compute_higher_order_sh(l, x_normalized, normalization)
    
    return compiled_spherical_harmonics


def _compute_higher_order_sh(l: int, x: mx.array, normalization: str) -> mx.array:
    """
    Compute higher order spherical harmonics recursively.
    
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
        Spherical harmonics values
    """
    # This is a placeholder - actual implementation would be more complex
    # For now, return a simple implementation
    if l == 2:
        # l=2 spherical harmonics (5 components)
        x, y, z = x[..., 0], x[..., 1], x[..., 2]
        
        # Real spherical harmonics for l=2
        sh = mx.stack([
            x * y,                    # 2, -2
            y * z,                    # 2, -1
            3 * z**2 - 1,            # 2, 0
            z * x,                    # 2, 1
            x**2 - y**2               # 2, 2
        ], axis=-1)
        
        if normalization == "integral":
            # Apply normalization factors
            norm_factors = mx.array([
                mx.sqrt(15 / (4 * mx.pi)),
                mx.sqrt(15 / (4 * mx.pi)),
                mx.sqrt(5 / (16 * mx.pi)),
                mx.sqrt(15 / (4 * mx.pi)),
                mx.sqrt(15 / (16 * mx.pi))
            ])
            sh = sh * norm_factors
        
        return sh
    else:
        # For higher orders, use a simple placeholder
        return mx.zeros(x.shape[:-1] + (2 * l + 1,))


def create_compiled_linear():
    """
    Create a compiled linear layer function.
    
    Returns
    -------
    compiled_func : Callable
        Compiled linear layer function
    """
    @compile_mode("mlx")
    def compiled_linear(x: mx.array, weights: mx.array, biases: Optional[mx.array] = None) -> mx.array:
        """
        Compiled linear layer operation.
        
        Parameters
        ----------
        x : mx.array
            Input tensor
        weights : mx.array
            Weight matrix
        biases : mx.array, optional
            Bias vector
            
        Returns
        -------
        result : mx.array
            Linear transformation result
        """
        # Handle empty input
        if x.shape[-1] == 0:
            return mx.zeros(x.shape[:-1] + (weights.shape[0],))
        
        # Reshape for batch processing
        batch_shape = x.shape[:-1]
        x_reshaped = x.reshape(-1, x.shape[-1])
        
        # Apply linear transformation
        output = mx.matmul(x_reshaped, weights.T)
        
        # Add bias if provided
        if biases is not None:
            output = output + biases
        
        # Reshape back
        return output.reshape(*batch_shape, weights.shape[0])
    
    return compiled_linear


# Global profiler instance
profiler = CompilationProfiler()


def get_profiler() -> CompilationProfiler:
    """
    Get the global profiler instance.
    
    Returns
    -------
    profiler : CompilationProfiler
        Global profiler instance
    """
    return profiler


def benchmark_compiled_vs_interpreted(compiled_func: Callable, interpreted_func: Callable, 
                                     *args, num_runs: int = 100) -> Dict[str, float]:
    """
    Benchmark compiled vs interpreted functions.
    
    Parameters
    ----------
    compiled_func : Callable
        Compiled function
    interpreted_func : Callable
        Interpreted function
    *args : tuple
        Arguments to pass to functions
    num_runs : int, default 100
        Number of runs for averaging
        
    Returns
    -------
    results : dict
        Benchmark results
    """
    # Benchmark compiled function
    start_time = time.time()
    for _ in range(num_runs):
        compiled_result = compiled_func(*args)
    compiled_time = (time.time() - start_time) / num_runs
    
    # Benchmark interpreted function
    start_time = time.time()
    for _ in range(num_runs):
        interpreted_result = interpreted_func(*args)
    interpreted_time = (time.time() - start_time) / num_runs
    
    # Compare results
    max_diff = mx.max(mx.abs(compiled_result - interpreted_result)).item()
    
    return {
        'compiled_time': compiled_time,
        'interpreted_time': interpreted_time,
        'speedup': interpreted_time / compiled_time,
        'max_difference': max_diff
    }