"""Argument tools for e3nn-mlx.

This module provides utilities for handling arguments in equivariant operations,
adapted from e3nn's _argtools.py for MLX.
"""
from typing import Optional, List, Union, Any
import warnings

import mlx.core as mx
import numpy as np

# Import Irreps locally to avoid circular imports
def _get_irreps_class():
    from e3nn_mlx.o3 import Irreps
    return Irreps


def _transform(dat, irreps_dat, rot_mat, translation: float = 0.0, output_transform_dtype: bool = False):
    """Transform ``dat`` by ``rot_mat`` and ``translation`` according to ``irreps_dat``.
    
    Parameters
    ----------
    dat : list or mx.array
        Data to transform
    irreps_dat : list
        List of irreps specifications for each data element
    rot_mat : mx.array
        3x3 rotation matrix
    translation : float or mx.array, optional
        Translation vector, by default 0.0
    output_transform_dtype : bool, optional
        Whether to output in transform dtype, by default False
        
    Returns
    -------
    list
        Transformed data
    """
    out = []
    transform_dtype = rot_mat.dtype
    translation = mx.array(translation, dtype=transform_dtype)
    
    if not isinstance(dat, (list, tuple)):
        dat = [dat]
    
    for irreps, a in zip(irreps_dat, dat):
        if output_transform_dtype:
            out_dtype = transform_dtype
        else:
            out_dtype = a.dtype
            
        if irreps is None:
            out.append(a.copy())
        elif irreps == "cartesian_points":
            translation = mx.array(translation, dtype=a.dtype)
            out.append((mx.array(a, dtype=transform_dtype) @ rot_mat.T + translation).astype(out_dtype))
        else:
            # For o3.Irreps
            Irreps = _get_irreps_class()
            if isinstance(irreps, Irreps):
                D = irreps.D_from_matrix(rot_mat)
                out.append((mx.array(a, dtype=transform_dtype) @ D.T).astype(out_dtype))
            else:
                # Assume it's a single irrep
                out.append((mx.array(a, dtype=transform_dtype) @ rot_mat.T).astype(out_dtype))
    
    return out


def _get_io_irreps(func, irreps_in=None, irreps_out=None):
    """Preprocess or, if not given, try to infer the I/O irreps for ``func``.
    
    Parameters
    ----------
    func : callable
        Function to get irreps for
    irreps_in : list or Irreps or str, optional
        Input irreps specification
    irreps_out : list or Irreps or str, optional
        Output irreps specification
        
    Returns
    -------
    tuple
        (irreps_in, irreps_out) as lists
    """
    SPECIAL_VALS = ["cartesian_points", None]
    Irreps = _get_irreps_class()

    if (irreps_in is None or irreps_out is None):
        # Try to infer from function attributes
        pass

    if irreps_in is None:
        if hasattr(func, "irreps_in"):
            irreps_in = func.irreps_in
        elif hasattr(func, "irreps_in1") and hasattr(func, "irreps_in2"):
            irreps_in = [func.irreps_in1, func.irreps_in2]
        else:
            raise ValueError(f"Cannot infer irreps_in for {func!r}; provide them explicitly")
    
    if irreps_out is None:
        if hasattr(func, "irreps_out"):
            irreps_out = func.irreps_out
        else:
            raise ValueError(f"Cannot infer irreps_out for {func!r}; provide them explicitly")

    # Normalize irreps_in to list format
    if isinstance(irreps_in, Irreps) or irreps_in in SPECIAL_VALS:
        irreps_in = [irreps_in]
    elif isinstance(irreps_in, list):
        irreps_in = [i if i in SPECIAL_VALS else Irreps(i) for i in irreps_in]
    else:
        if isinstance(irreps_in, tuple) and not isinstance(irreps_in, Irreps):
            warnings.warn(
                f"Module {func} had irreps_in of type tuple but not Irreps; ambiguous whether the tuple should be interpreted "
                f"as a tuple representing a single Irreps or a tuple of objects each to be converted to Irreps. Assuming the "
                f"former. If the latter, use a list."
            )
        irreps_in = [Irreps(irreps_in)]

    # Normalize irreps_out to list format
    if isinstance(irreps_out, Irreps) or irreps_out in SPECIAL_VALS:
        irreps_out = [irreps_out]
    elif isinstance(irreps_out, list):
        irreps_out = [i if i in SPECIAL_VALS else Irreps(i) for i in irreps_out]
    else:
        if isinstance(irreps_out, tuple) and not isinstance(irreps_out, Irreps):
            warnings.warn(
                f"Module {func} had irreps_out of type tuple but not Irreps; ambiguous whether the tuple should be "
                f"interpreted as a tuple representing a single Irreps or a tuple of objects each to be converted to Irreps. "
                f"Assuming the former. If the latter, use a list."
            )
        irreps_out = [Irreps(irreps_out)]

    return irreps_in, irreps_out


def _get_args_in(func, args_in=None, irreps_in=None, irreps_out=None):
    """Get input arguments for function testing.
    
    Parameters
    ----------
    func : callable
        Function to get arguments for
    args_in : list, optional
        Input arguments
    irreps_in : list or Irreps, optional
        Input irreps specification
    irreps_out : list or Irreps, optional
        Output irreps specification
        
    Returns
    -------
    tuple
        (args_in, irreps_in, irreps_out)
    """
    irreps_in, irreps_out = _get_io_irreps(func, irreps_in=irreps_in, irreps_out=irreps_out)
    if args_in is None:
        args_in = _rand_args(irreps_in)
    assert len(args_in) == len(irreps_in), "irreps_in and args_in don't match in length"
    return args_in, irreps_in, irreps_out


def _rand_args(irreps_in, batch_size: Optional[int] = None):
    """Generate random arguments for testing.
    
    Parameters
    ----------
    irreps_in : list
        List of irreps specifications
    batch_size : int, optional
        Batch size for generated data
        
    Returns
    -------
    list
        List of random arguments
    """
    Irreps = _get_irreps_class()
    
    if not all((isinstance(i, Irreps) or i == "cartesian_points") for i in irreps_in):
        raise ValueError(
            "Random arguments cannot be generated when argument types besides Irreps and `'cartesian_points'` are specified; "
            "provide explicit ``args_in``"
        )
    if batch_size is None:
        # Generate random args with random size batch dim between 1 and 4:
        batch_size = np.random.randint(1, 5)
    
    args_in = []
    for irreps in irreps_in:
        if irreps == "cartesian_points":
            args_in.append(mx.random.normal((batch_size, 3)))
        else:
            args_in.append(irreps.randn(batch_size, -1))
    
    return args_in


def _get_device(mod) -> mx.Device:
    """Get the device of a module.
    
    Parameters
    ----------
    mod : callable
        Module to get device for
        
    Returns
    -------
    mx.Device
        Device of the module
    """
    # For MLX, we need to check if the module has parameters
    # This is a simplified implementation
    if hasattr(mod, 'parameters'):
        try:
            params = list(mod.parameters())
            if params:
                return params[0].device
        except (AttributeError, StopIteration):
            pass
    
    # Default to CPU/GPU based on availability
    if mx.default_device() is not None:
        return mx.default_device()
    return mx.cpu()


def _get_floating_dtype(mod) -> mx.Dtype:
    """Guess floating dtype for module.

    Assumes no mixed precision.

    Parameters
    ----------
    mod : callable
        Module to get dtype for
        
    Returns
    -------
    mx.Dtype
        Floating dtype of the module
    """
    # For MLX, we need to check if the module has parameters
    if hasattr(mod, 'parameters'):
        try:
            for param in mod.parameters():
                if mx.issubdtype(param.dtype, mx.floating):
                    return param.dtype
        except AttributeError:
            pass
    
    # Default to float32
    return mx.float32


def _to_device_dtype(args, device=None, dtype=None):
    """Move arguments to device and convert dtype.
    
    Parameters
    ----------
    args : mx.array or list or tuple or dict
        Arguments to move
    device : mx.Device, optional
        Device to move to
    dtype : mx.Dtype, optional
        Dtype to convert to
        
    Returns
    -------
    mx.array or list or tuple or dict
        Arguments moved to device and converted dtype
    """
    kwargs = {}
    if device is not None:
        kwargs["device"] = device
    if dtype is not None:
        kwargs["dtype"] = dtype

    if isinstance(args, mx.array):
        if mx.issubdtype(args.dtype, mx.floating):
            # Only convert dtypes of floating tensors
            return args.astype(dtype) if dtype is not None else args
        else:
            return args
    elif isinstance(args, tuple):
        return tuple(_to_device_dtype(e, **kwargs) for e in args)
    elif isinstance(args, list):
        return [_to_device_dtype(e, **kwargs) for e in args]
    elif isinstance(args, dict):
        return {k: _to_device_dtype(v, **kwargs) for k, v in args.items()}
    else:
        raise TypeError("Only (nested) dict/tuple/lists of arrays can be moved to a device/dtype.")


def _make_tracing_inputs(mod, n: int = 1):
    """Default tracing inputs generator for modules.
    
    Parameters
    ----------
    mod : callable
        Module to generate inputs for
    n : int, optional
        Number of input sets to generate
        
    Returns
    -------
    list
        List of input dictionaries
    """
    irreps_in, _ = _get_io_irreps(mod)
    device = _get_device(mod)
    dtype = _get_floating_dtype(mod)
    
    inputs = []
    for _ in range(n):
        args = _rand_args(irreps_in)
        args = _to_device_dtype(args, device, dtype)
        inputs.append({"forward": tuple(args)})
    
    return inputs