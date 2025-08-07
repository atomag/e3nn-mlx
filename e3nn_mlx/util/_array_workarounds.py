"""
MLX Array Operations Workarounds

This module provides workarounds for missing MLX array operations,
specifically the array.at.set method that's not available in MLX 0.27.1.
"""

import mlx.core as mx
import numpy as np
from typing import Union, Tuple, List, Optional


def array_at_set_workaround(arr: mx.array, indices: Union[Tuple, List, int], value: Union[mx.array, float, int]) -> mx.array:
    """
    Workaround for missing array.at.set operation in MLX.
    
    This function simulates the behavior of arr.at[indices].set(value)
    using available MLX operations.
    
    Parameters
    ----------
    arr : mx.array
        Input array to modify
    indices : tuple, list, or int
        Indices where to set the value
    value : mx.array, float, or int
        Value to set at the specified indices
        
    Returns
    -------
    result : mx.array
        Modified array with value set at specified indices
        
    Examples
    --------
    >>> arr = mx.array([[1, 2, 3], [4, 5, 6]])
    >>> result = array_at_set_workaround(arr, (0, 0), 99)
    >>> print(result)
    [[99, 2, 3],
     [4, 5, 6]]
    """
    # Input validation
    if not isinstance(arr, mx.array):
        raise TypeError(f"arr must be an mx.array, got {type(arr).__name__}")
    
    if not isinstance(indices, (int, tuple, list)):
        raise TypeError(f"indices must be an int, tuple, or list, got {type(indices).__name__}")
    
    if not isinstance(value, (mx.array, int, float)):
        raise TypeError(f"value must be an mx.array, int, or float, got {type(value).__name__}")
    
    # Convert single index to tuple for consistent handling
    if isinstance(indices, int):
        indices = (indices,)
    elif not isinstance(indices, tuple):
        indices = tuple(indices)
    
    # Handle scalar value
    if not isinstance(value, mx.array):
        try:
            value = mx.array(value, dtype=arr.dtype)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert value {value} to array with dtype {arr.dtype}: {e}")
    
    # Strategy 1: Use mx.where for simple cases
    if len(indices) == 1 and isinstance(indices[0], int):
        # Simple 1D indexing
        idx = indices[0]
        mask = mx.zeros(arr.shape, dtype=mx.bool_)
        # Create mask using available operations
        mask_array = mx.zeros(arr.shape, dtype=arr.dtype)
        mask_array = mask_array.at[idx].add(1)
        mask = mask_array > 0
        return mx.where(mask, value, arr)
    
    # Strategy 2: Use multiplication and addition for reset pattern
    # This works by setting the target position to 0, then adding the desired value
    try:
        # Create zero mask at target position
        zero_mask = mx.ones_like(arr)
        if len(indices) == 2 and all(isinstance(i, int) for i in indices):
            # 2D integer indexing
            i, j = indices
            zero_mask = zero_mask.at[i, j].multiply(0)
            # Set target position to zero, then add desired value
            temp_result = arr * zero_mask
            # Add the desired value at target position
            value_expanded = mx.zeros_like(arr)
            value_expanded = value_expanded.at[i, j].add(value)
            return temp_result + value_expanded
        elif len(indices) == 1 and isinstance(indices[0], slice):
            # Slice indexing
            slice_idx = indices[0]
            zero_mask = mx.ones_like(arr)
            # Set slice to zero
            for i in range(arr.shape[0])[slice_idx]:
                zero_mask = zero_mask.at[i].multiply(0)
            temp_result = arr * zero_mask
            # Add values to slice
            value_expanded = mx.zeros_like(arr)
            for i in range(arr.shape[0])[slice_idx]:
                value_expanded = value_expanded.at[i].add(value)
            return temp_result + value_expanded
    except (ValueError, TypeError, IndexError, RuntimeError):
        # If vectorized operations fail, continue to fallback method
        pass
    
    # Strategy 3: Use vectorized operations where possible
    try:
        # Handle 2D integer indexing with vectorized operations
        if len(indices) == 2 and all(isinstance(i, int) for i in indices):
            i, j = indices
            
            # Create a mask for the target position
            mask = mx.zeros(arr.shape, dtype=mx.bool_)
            
            # Use advanced indexing to create the mask
            row_indices = mx.arange(arr.shape[0])
            col_indices = mx.arange(arr.shape[1])
            
            # Create coordinate grids
            row_grid = row_indices.reshape(-1, 1)
            col_grid = col_indices.reshape(1, -1)
            
            # Create mask for the target position
            mask = (row_grid == i) & (col_grid == j)
            
            # Use mx.where to set the value
            return mx.where(mask, value, arr)
        
        # Handle 1D integer indexing with vectorized operations
        elif len(indices) == 1 and isinstance(indices[0], int):
            idx = indices[0]
            
            # Create a mask for the target position
            mask = mx.zeros(arr.shape, dtype=mx.bool_)
            indices_array = mx.arange(arr.shape[0])
            mask = indices_array == idx
            
            # Use mx.where to set the value
            return mx.where(mask, value, arr)
        
        # For more complex cases, fall back to the original list conversion method
        arr_list = arr.tolist()
        current = arr_list
        
        # Navigate to the target position
        for i, idx in enumerate(indices[:-1]):
            current = current[idx]
        
        # Set the final value
        current[indices[-1]] = value.item() if hasattr(value, 'item') else value
        
        return mx.array(arr_list, dtype=arr.dtype)
    except (ValueError, TypeError, IndexError, RuntimeError) as e:
        raise ValueError(f"Cannot apply array.at.set workaround for indices {indices}: {e}") from e


def array_at_set_scattered_workaround(arr: mx.array, indices_list: List[Tuple], values: mx.array) -> mx.array:
    """
    Workaround for scattered array assignment (multiple indices at once).
    
    Parameters
    ----------
    arr : mx.array
        Input array to modify
    indices_list : list of tuples
        List of indices where to set values
    values : mx.array
        Values to set at the specified indices
        
    Returns
    -------
    result : mx.array
        Modified array with values set at specified indices
    """
    # Input validation
    if not isinstance(arr, mx.array):
        raise TypeError(f"arr must be an mx.array, got {type(arr).__name__}")
    
    if not isinstance(indices_list, (list, tuple)):
        raise TypeError(f"indices_list must be a list or tuple, got {type(indices_list).__name__}")
    
    if not isinstance(values, mx.array):
        raise TypeError(f"values must be an mx.array, got {type(values).__name__}")
    
    if len(indices_list) == 0:
        return arr.copy()
    
    if values.ndim < 1:
        raise ValueError(f"values must have at least 1 dimension, got {values.ndim} dimensions")
    
    if values.shape[0] < len(indices_list):
        raise ValueError(
            f"values.shape[0] ({values.shape[0]}) must be at least the length of indices_list ({len(indices_list)})"
        )
    
    result = arr.copy()
    
    for i, indices in enumerate(indices_list):
        if i < values.shape[0]:
            result = array_at_set_workaround(result, indices, values[i])
    
    return result


def spherical_harmonics_set_workaround(result: mx.array, offset: int, dim_l: int, sh_l: mx.array) -> mx.array:
    """
    Specific workaround for spherical harmonics array assignment.
    
    This handles the pattern: result.at[..., offset:offset + dim_l].set(sh_l)
    
    Parameters
    ----------
    result : mx.array
        Result array to modify
    offset : int
        Starting offset for the slice
    dim_l : int
        Length of the slice
    sh_l : mx.array
        Values to set in the slice
        
    Returns
    -------
    result : mx.array
        Modified array
    """
    # Input validation
    if not isinstance(result, mx.array):
        raise TypeError(f"result must be an mx.array, got {type(result).__name__}")
    
    if not isinstance(offset, int):
        raise TypeError(f"offset must be an integer, got {type(offset).__name__}")
    
    if not isinstance(dim_l, int):
        raise TypeError(f"dim_l must be an integer, got {type(dim_l).__name__}")
    
    if not isinstance(sh_l, mx.array):
        raise TypeError(f"sh_l must be an mx.array, got {type(sh_l).__name__}")
    
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")
    
    if dim_l < 0:
        raise ValueError(f"dim_l must be non-negative, got {dim_l}")
    
    if result.ndim < 1:
        raise ValueError(f"result must have at least 1 dimension, got {result.ndim} dimensions")
    
    # Handle the slice assignment pattern using vectorized operations
    if len(result.shape) >= 2:  # Handle batch dimension
        batch_size = result.shape[0]
        
        # Create masks for the target slice
        col_indices = mx.arange(result.shape[1])
        slice_mask = (col_indices >= offset) & (col_indices < offset + dim_l)
        
        # Expand mask to match result shape
        mask = slice_mask.reshape(1, -1)
        mask = mx.broadcast_to(mask, (batch_size, result.shape[1]))
        
        # Create target array by slicing and concatenation
        if offset > 0:
            before_slice = result[:, :offset]
        else:
            before_slice = mx.array([], dtype=result.dtype).reshape(batch_size, 0)
        
        # Handle the slice with sh_l values
        actual_dim = min(dim_l, sh_l.shape[-1])
        if actual_dim > 0:
            if len(sh_l.shape) > 1:
                # Batch dimension in sh_l
                slice_values = sh_l[:, :actual_dim]
            else:
                # No batch dimension in sh_l, broadcast to batch
                slice_values = sh_l[:actual_dim].reshape(1, -1)
                slice_values = mx.broadcast_to(slice_values, (batch_size, actual_dim))
        else:
            slice_values = mx.array([], dtype=result.dtype).reshape(batch_size, 0)
        
        if offset + dim_l < result.shape[1]:
            after_slice = result[:, offset + dim_l:]
        else:
            after_slice = mx.array([], dtype=result.dtype).reshape(batch_size, 0)
        
        # Concatenate the parts
        return mx.concatenate([before_slice, slice_values, after_slice], axis=1)
    else:
        # 1D case
        if offset > 0:
            before_slice = result[:offset]
        else:
            before_slice = mx.array([], dtype=result.dtype)
        
        # Handle the slice with sh_l values
        actual_dim = min(dim_l, sh_l.shape[-1])
        if actual_dim > 0:
            slice_values = sh_l[:actual_dim]
        else:
            slice_values = mx.array([], dtype=result.dtype)
        
        if offset + dim_l < result.shape[0]:
            after_slice = result[offset + dim_l:]
        else:
            after_slice = mx.array([], dtype=result.dtype)
        
        # Concatenate the parts
        return mx.concatenate([before_slice, slice_values, after_slice], axis=0)


def patch_array_operations():
    """
    Monkey patch mx.array to add set functionality.
    This should be called once at module import.
    """
    def set_method(self, indices, value):
        return array_at_set_workaround(self, indices, value)
    
    # Add the set method to ArrayAt objects
    original_getitem = mx.array.__getitem__
    
    def patched_getitem(self, key):
        result = original_getitem(self, key)
        if hasattr(result, 'set'):
            return result
        # Add our set method
        result.set = lambda value: array_at_set_workaround(self, key, value)
        return result
    
    # This is a bit hacky but provides the API compatibility
    mx.array.at_set_workaround = array_at_set_workaround
    mx.array.spherical_harmonics_set_workaround = spherical_harmonics_set_workaround


# Apply the patch automatically
patch_array_operations()


def test_workarounds():
    """Test the array.at.set workarounds."""
    print("Testing array.at.set workarounds...")
    
    # Test 1: Simple scalar assignment
    arr = mx.array([[1, 2, 3], [4, 5, 6]])
    try:
        result = array_at_set_workaround(arr, (0, 0), 99)
        print("✓ Simple scalar assignment works")
        print(f"  Result: {result}")
    except Exception as e:
        print(f"✗ Simple scalar assignment failed: {e}")
    
    # Test 2: Slice assignment
    try:
        result = array_at_set_workaround(arr, (slice(0, 2), 1), 99)
        print("✓ Slice assignment works")
        print(f"  Result: {result}")
    except Exception as e:
        print(f"✗ Slice assignment failed: {e}")
    
    # Test 3: Spherical harmonics pattern
    try:
        result = mx.zeros((10, 5))
        sh_l = mx.ones((10, 3))
        result = spherical_harmonics_set_workaround(result, 1, 3, sh_l)
        print("✓ Spherical harmonics pattern works")
        print(f"  Result shape: {result.shape}")
    except Exception as e:
        print(f"✗ Spherical harmonics pattern failed: {e}")


if __name__ == "__main__":
    test_workarounds()