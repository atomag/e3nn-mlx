"""
Default type and device utilities (MLX).

This mirrors e3nn.util.default_type for PyTorch, adapted to MLX and the
e3nn-mlx context system.
"""

from typing import Optional, Tuple

import mlx.core as mx

from ._context import (
    get_default_dtype,
    get_default_device,
    set_default_dtype,
    set_default_device,
    default_dtype as context_default_dtype,
    default_device as context_default_device,
)


def mlx_get_default_dtype() -> mx.Dtype:
    """Return the current default floating dtype.

    Resolves e3nn-mlx context default dtype first; otherwise falls back to
    MLX's de facto default (float32).
    """
    dtype = get_default_dtype()
    return dtype if dtype is not None else mx.float32


def mlx_get_default_device() -> mx.Device:
    """Return the current default device.

    Resolves e3nn-mlx context default device first; otherwise returns
    MLX default device, or CPU if unset.
    """
    dev = get_default_device()
    if dev is not None:
        return dev
    d = mx.default_device()
    return d if d is not None else mx.cpu()


def explicit_default_types(
    dtype: Optional[mx.Dtype], device: Optional[mx.Device]
) -> Tuple[mx.Dtype, mx.Device]:
    """Resolve dtype and device with explicit defaults.

    If a value is None, use the current defaults from the e3nn-mlx context or
    MLX fallbacks.
    """
    if dtype is None:
        dtype = mlx_get_default_dtype()
    if device is None:
        device = mlx_get_default_device()
    return dtype, device


# Convenience re-exports for context management
def set_default_dtype_and_device(dtype: Optional[mx.Dtype] = None, device: Optional[mx.Device] = None) -> None:
    """Set global defaults in the e3nn-mlx context (optional per argument)."""
    if dtype is not None:
        set_default_dtype(dtype)
    if device is not None:
        set_default_device(device)


# Context manager aliases for convenience (match e3nn spirit)
default_dtype = context_default_dtype
default_device = context_default_device


class add_type_kwargs:
    """Decorator to add dtype/device kwargs and apply them via context.

    Example:
    >>> @add_type_kwargs()
    ... def rand_matrix(*shape, dtype=None, device=None):
    ...     ...
    """

    _DOC_NOTE = (
        "\n- dtype and device keyword arguments will be passed to e3nn_mlx.util.default_type context managers\n"
    )

    def __init__(self, dtype=None, device=None) -> None:
        self.dtype = dtype
        self.device = device

    def __call__(self, f):
        def wrapper(*args, dtype=None, device=None, **kwargs):
            # resolve provided or default decorator args
            eff_dtype = dtype if dtype is not None else self.dtype
            eff_device = device if device is not None else self.device
            # Apply context managers if provided
            if eff_dtype is not None and eff_device is not None:
                with context_default_dtype(eff_dtype), context_default_device(eff_device):
                    return f(*args, **kwargs)
            elif eff_dtype is not None:
                with context_default_dtype(eff_dtype):
                    return f(*args, **kwargs)
            elif eff_device is not None:
                with context_default_device(eff_device):
                    return f(*args, **kwargs)
            else:
                return f(*args, **kwargs)

        # Augment docstring
        if getattr(f, "__doc__", None):
            if not f.__doc__.endswith("\n"):
                f.__doc__ += "\n"
            wrapper.__doc__ = f.__doc__ + self._DOC_NOTE
        return wrapper
