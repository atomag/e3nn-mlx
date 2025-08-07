"""Context management utilities for e3nn-mlx.

This module provides context managers for managing default device and dtype,
adapted from e3nn's _context.py for MLX.
"""
from contextlib import contextmanager
from typing import Optional

import mlx.core as mx


# Global context variables
_default_dtype = None
_default_device = None


@contextmanager
def default_dtype(dtype: mx.Dtype):
    """Context manager for setting the default dtype.
    
    Parameters
    ----------
    dtype : mx.Dtype
        The default dtype to use within the context
    """
    global _default_dtype
    old_dtype = _default_dtype
    _default_dtype = dtype
    try:
        yield
    finally:
        _default_dtype = old_dtype


@contextmanager
def default_device(device: mx.Device):
    """Context manager for setting the default device.
    
    Parameters
    ----------
    device : mx.Device
        The default device to use within the context
    """
    global _default_device
    old_device = _default_device
    _default_device = device
    try:
        yield
    finally:
        _default_device = old_device


def get_default_dtype() -> Optional[mx.Dtype]:
    """Get the current default dtype.
    
    Returns
    -------
    mx.Dtype or None
        The current default dtype, or None if not set
    """
    return _default_dtype


def get_default_device() -> Optional[mx.Device]:
    """Get the current default device.
    
    Returns
    -------
    mx.Device or None
        The current default device, or None if not set
    """
    return _default_device


def set_default_dtype(dtype: mx.Dtype):
    """Set the default dtype.
    
    Parameters
    ----------
    dtype : mx.Dtype
        The default dtype to set
    """
    global _default_dtype
    _default_dtype = dtype


def set_default_device(device: mx.Device):
    """Set the default device.
    
    Parameters
    ----------
    device : mx.Device
        The default device to set
    """
    global _default_device
    _default_device = device


@contextmanager
def explicit_default_types():
    """Context manager that explicitly sets default types.
    
    This is useful for ensuring consistent behavior across different
    environments and for debugging type-related issues.
    """
    old_dtype = get_default_dtype()
    old_device = get_default_device()
    
    # Set explicit defaults
    set_default_dtype(mx.float32)
    set_default_device(mx.default_device() or mx.cpu())
    
    try:
        yield
    finally:
        # Restore old values
        set_default_dtype(old_dtype)
        set_default_device(old_device)


def resolve_dtype(dtype: Optional[mx.Dtype] = None) -> mx.Dtype:
    """Resolve the dtype to use, considering context defaults.
    
    Parameters
    ----------
    dtype : mx.Dtype, optional
        The dtype to use, or None to use default
        
    Returns
    -------
    mx.Dtype
        The resolved dtype
    """
    if dtype is not None:
        return dtype
    if _default_dtype is not None:
        return _default_dtype
    return mx.float32


def resolve_device(device: Optional[mx.Device] = None) -> mx.Device:
    """Resolve the device to use, considering context defaults.
    
    Parameters
    ----------
    device : mx.Device, optional
        The device to use, or None to use default
        
    Returns
    -------
    mx.Device
        The resolved device
    """
    if device is not None:
        return device
    if _default_device is not None:
        return _default_device
    return mx.default_device() or mx.cpu()