"""
Validation utilities for e3nn-mlx

This module provides common validation patterns used throughout the e3nn-mlx library.
"""

import mlx.core as mx
from typing import Union, List, Tuple, Any, Optional


def validate_type(value: Any, expected_type: Union[type, Tuple[type, ...]], name: str) -> None:
    """
    Validate that a value is of the expected type.
    
    Parameters
    ----------
    value : Any
        The value to validate
    expected_type : type or tuple of types
        The expected type(s)
    name : str
        The name of the parameter being validated (for error messages)
        
    Raises
    ------
    TypeError
        If the value is not of the expected type
    """
    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            type_names = [t.__name__ for t in expected_type]
            expected_str = ", ".join(type_names[:-1]) + f", or {type_names[-1]}"
        else:
            expected_str = expected_type.__name__
        
        raise TypeError(f"{name} must be {expected_str}, got {type(value).__name__}")


def validate_positive_number(value: Union[int, float], name: str, allow_zero: bool = False) -> None:
    """
    Validate that a value is a positive number.
    
    Parameters
    ----------
    value : int or float
        The value to validate
    name : str
        The name of the parameter being validated (for error messages)
    allow_zero : bool, default False
        Whether to allow zero as a valid value
        
    Raises
    ------
    TypeError
        If the value is not a number
    ValueError
        If the value is not positive (or non-negative if allow_zero=True)
    """
    validate_type(value, (int, float), name)
    
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
    else:
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")


def validate_range(value: Union[int, float], name: str, min_val: Union[int, float], max_val: Union[int, float], 
                  inclusive_min: bool = True, inclusive_max: bool = True) -> None:
    """
    Validate that a value is within a specified range.
    
    Parameters
    ----------
    value : int or float
        The value to validate
    name : str
        The name of the parameter being validated (for error messages)
    min_val : int or float
        Minimum allowed value
    max_val : int or float
        Maximum allowed value
    inclusive_min : bool, default True
        Whether the minimum value is inclusive
    inclusive_max : bool, default True
        Whether the maximum value is inclusive
        
    Raises
    ------
    TypeError
        If the value is not a number
    ValueError
        If the value is outside the specified range
    """
    validate_type(value, (int, float), name)
    
    if inclusive_min and inclusive_max:
        if not (min_val <= value <= max_val):
            raise ValueError(f"{name} must be between {min_val} and {max_val} (inclusive), got {value}")
    elif inclusive_min and not inclusive_max:
        if not (min_val <= value < max_val):
            raise ValueError(f"{name} must be between {min_val} (inclusive) and {max_val} (exclusive), got {value}")
    elif not inclusive_min and inclusive_max:
        if not (min_val < value <= max_val):
            raise ValueError(f"{name} must be between {min_val} (exclusive) and {max_val} (inclusive), got {value}")
    else:
        if not (min_val < value < max_val):
            raise ValueError(f"{name} must be between {min_val} and {max_val} (exclusive), got {value}")


def validate_array_like(value: Any, name: str, allow_none: bool = False) -> None:
    """
    Validate that a value is array-like (mx.array, list, tuple).
    
    Parameters
    ----------
    value : Any
        The value to validate
    name : str
        The name of the parameter being validated (for error messages)
    allow_none : bool, default False
        Whether to allow None as a valid value
        
    Raises
    ------
    TypeError
        If the value is not array-like
    """
    if allow_none and value is None:
        return
    
    if not isinstance(value, (mx.array, list, tuple)):
        raise TypeError(f"{name} must be an array-like object (mx.array, list, or tuple), got {type(value).__name__}")


def validate_array_dimensions(arr: mx.array, name: str, min_dims: int = 1, max_dims: Optional[int] = None) -> None:
    """
    Validate that an array has the expected number of dimensions.
    
    Parameters
    ----------
    arr : mx.array
        The array to validate
    name : str
        The name of the parameter being validated (for error messages)
    min_dims : int, default 1
        Minimum number of dimensions
    max_dims : int or None, default None
        Maximum number of dimensions (None for no maximum)
        
    Raises
    ------
    TypeError
        If the input is not an mx.array
    ValueError
        If the array doesn't have the expected number of dimensions
    """
    validate_type(arr, mx.array, name)
    
    if arr.ndim < min_dims:
        raise ValueError(f"{name} must have at least {min_dims} dimension(s), got {arr.ndim}")
    
    if max_dims is not None and arr.ndim > max_dims:
        raise ValueError(f"{name} must have at most {max_dims} dimension(s), got {arr.ndim}")


def validate_array_shape(arr: mx.array, name: str, expected_shape: Optional[Union[int, Tuple[int, ...]]] = None, 
                       expected_dim: Optional[int] = None) -> None:
    """
    Validate that an array has the expected shape or dimension.
    
    Parameters
    ----------
    arr : mx.array
        The array to validate
    name : str
        The name of the parameter being validated (for error messages)
    expected_shape : int or tuple of int or None, default None
        Expected shape (if None, only validates expected_dim)
    expected_dim : int or None, default None
        Expected last dimension (if None, only validates expected_shape)
        
    Raises
    ------
    TypeError
        If the input is not an mx.array
    ValueError
        If the array doesn't have the expected shape or dimension
    """
    validate_type(arr, mx.array, name)
    
    if expected_shape is not None:
        if isinstance(expected_shape, int):
            expected_shape = (expected_shape,)
        
        if len(expected_shape) != len(arr.shape):
            raise ValueError(
                f"{name} expected to have {len(expected_shape)} dimension(s), "
                f"got {len(arr.shape)} dimension(s) with shape {arr.shape}"
            )
        
        for i, (expected, actual) in enumerate(zip(expected_shape, arr.shape)):
            if expected != actual:
                raise ValueError(
                    f"{name} dimension {i} expected to be {expected}, "
                    f"got {actual} (full shape: {arr.shape})"
                )
    
    if expected_dim is not None:
        if arr.shape[-1] != expected_dim:
            raise ValueError(
                f"{name} last dimension must be {expected_dim}, "
                f"got {arr.shape[-1]} (full shape: {arr.shape})"
            )


def validate_string_choice(value: str, name: str, valid_choices: List[str], case_sensitive: bool = True) -> None:
    """
    Validate that a string value is one of the valid choices.
    
    Parameters
    ----------
    value : str
        The value to validate
    name : str
        The name of the parameter being validated (for error messages)
    valid_choices : list of str
        List of valid choices
    case_sensitive : bool, default True
        Whether the comparison should be case-sensitive
        
    Raises
    ------
    TypeError
        If the value is not a string
    ValueError
        If the value is not one of the valid choices
    """
    validate_type(value, str, name)
    
    if case_sensitive:
        if value not in valid_choices:
            raise ValueError(f"{name} must be one of {valid_choices}, got '{value}'")
    else:
        value_lower = value.lower()
        valid_choices_lower = [choice.lower() for choice in valid_choices]
        if value_lower not in valid_choices_lower:
            raise ValueError(f"{name} must be one of {valid_choices} (case-insensitive), got '{value}'")


def validate_list_length(value: List[Any], name: str, expected_length: int, allow_empty: bool = False) -> None:
    """
    Validate that a list has the expected length.
    
    Parameters
    ----------
    value : list
        The list to validate
    name : str
        The name of the parameter being validated (for error messages)
    expected_length : int
        Expected length of the list
    allow_empty : bool, default False
        Whether to allow empty lists when expected_length is 0
        
    Raises
    ------
    TypeError
        If the value is not a list
    ValueError
        If the list doesn't have the expected length
    """
    validate_type(value, list, name)
    
    if not allow_empty and expected_length == 0 and len(value) > 0:
        raise ValueError(f"{name} must be empty, got {len(value)} elements")
    
    if expected_length > 0 and len(value) != expected_length:
        raise ValueError(f"{name} must have {expected_length} element(s), got {len(value)}")


def validate_boolean(value: Any, name: str) -> None:
    """
    Validate that a value is a boolean.
    
    Parameters
    ----------
    value : Any
        The value to validate
    name : str
        The name of the parameter being validated (for error messages)
        
    Raises
    ------
    TypeError
        If the value is not a boolean
    """
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean, got {type(value).__name__}")


def validate_not_none(value: Any, name: str) -> None:
    """
    Validate that a value is not None.
    
    Parameters
    ----------
    value : Any
        The value to validate
    name : str
        The name of the parameter being validated (for error messages)
        
    Raises
    ------
    ValueError
        If the value is None
    """
    if value is None:
        raise ValueError(f"{name} cannot be None")


class ValidationError(Exception):
    """Base class for validation errors."""
    pass


class InputValidationError(ValidationError):
    """Error raised when input validation fails."""
    pass


class ShapeValidationError(ValidationError):
    """Error raised when shape validation fails."""
    pass


class TypeValidationError(ValidationError):
    """Error raised when type validation fails."""
    pass


class ValueValidationError(ValidationError):
    """Error raised when value validation fails."""
    pass