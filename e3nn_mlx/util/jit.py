"""JIT compilation utilities for e3nn-mlx.

This module provides compilation functionality similar to e3nn's JIT system,
adapted for MLX's compilation model.
"""
import copy
import inspect
import warnings
from contextlib import contextmanager
from functools import wraps
from typing import Optional, Callable, Dict, Any, List, Tuple

import mlx.core as mx


_COMPILE_MODE_ATTR = "__e3nn_compile_mode__"
_VALID_MODES = ("trace", "script", "unsupported", None)
_MAKE_TRACING_INPUTS = "_make_tracing_inputs"


def compile_mode(mode: str):
    """Decorator to set the compile mode of a module.

    Parameters
    ----------
    mode : str
        'script', 'trace', or None
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"Invalid compile mode: {mode}. Must be one of {_VALID_MODES}")

    def decorator(obj):
        if not (inspect.isclass(obj) and hasattr(obj, '__call__')):
            raise TypeError("@e3nn_mlx.util.jit.compile_mode can only decorate callable classes")
        setattr(obj, _COMPILE_MODE_ATTR, mode)
        return obj

    return decorator


def get_compile_mode(mod) -> str:
    """Get the compilation mode of a module.

    Parameters
    ----------
    mod : callable
        The module or function to check

    Returns
    -------
    str
        'script', 'trace', or None if the module was not decorated with @compile_mode
    """
    if hasattr(mod, _COMPILE_MODE_ATTR):
        mode = getattr(mod, _COMPILE_MODE_ATTR)
    else:
        mode = getattr(type(mod), _COMPILE_MODE_ATTR, None)
    
    if mode not in _VALID_MODES:
        raise ValueError(f"Invalid compile mode `{mode}`")
    return mode


def compile(
    mod,
    n_trace_checks: int = 1,
    script_options: dict = None,
    trace_options: dict = None,
    in_place: bool = True,
    recurse: bool = True,
):
    """Recursively compile a module and all submodules according to their decorators.

    (Sub)modules without decorators will be unaffected.

    Parameters
    ----------
    mod : callable
        The module to compile. The module will have its submodules compiled replaced in-place.
    n_trace_checks : int, default = 1
        How many random example inputs to generate when tracing a module. Must be at least one.
    script_options : dict, default = {}
        Extra kwargs for MLX compilation.
    trace_options : dict, default = {}
        Extra kwargs for MLX tracing.
    in_place : bool, default True
        Whether to insert the recursively compiled submodules in-place, or do a deepcopy first.
    recurse : bool, default True
        Whether to recurse through the module's children before passing the parent to compilation

    Returns
    -------
    callable
        Returns the compiled module.
    """
    script_options = script_options or {}
    trace_options = trace_options or {}

    mode = get_compile_mode(mod)
    if mode == "unsupported":
        raise NotImplementedError(f"{type(mod).__name__} does not support MLX compilation")

    if not in_place:
        mod = copy.deepcopy(mod)
    
    assert n_trace_checks >= 1

    if recurse:
        # == recurse to children ==
        # This allows us to trace compile submodules of modules we are going to compile
        for submod_name, submod in getattr(mod, '__dict__', {}).items():
            if hasattr(submod, _COMPILE_MODE_ATTR) or hasattr(type(submod), _COMPILE_MODE_ATTR):
                setattr(
                    mod,
                    submod_name,
                    compile(
                        submod,
                        n_trace_checks=n_trace_checks,
                        script_options=script_options,
                        trace_options=trace_options,
                        in_place=True,  # since we deepcopied the module above, we can do inplace
                        recurse=recurse,  # always true in this branch
                    ),
                )

    # == Compile this module now ==
    if mode == "script":
        # For MLX, we use mx.compile for script mode
        mod = mx.compile(mod, **script_options)
    elif mode == "trace":
        # For trace mode, we need tracing inputs
        check_inputs = get_tracing_inputs(
            mod,
            n_trace_checks,
        )
        assert len(check_inputs) >= 1, "Must have at least one tracing input."
        
        # For MLX tracing, we'll use the first input for compilation
        # MLX doesn't have direct equivalent of torch.jit.trace_module
        # We'll use mx.compile with example inputs
        example_input = check_inputs[0].get('forward', ())
        if isinstance(example_input, tuple) and len(example_input) == 1:
            example_input = example_input[0]
        
        mod = mx.compile(mod, **trace_options)
    
    return mod


def get_tracing_inputs(
    mod, n: int = 1, device: Optional[mx.Device] = None, dtype: Optional[mx.Dtype] = None
):
    """Get random tracing inputs for ``mod``.

    First checks if ``mod`` has a ``_make_tracing_inputs`` method. If so, calls it with ``n`` as the single argument and
    returns its results.

    Otherwise, attempts to infer the input signature of the module using ``e3nn_mlx.util._argtools._get_io_irreps``.

    Parameters
    ----------
    mod : callable
        The module to generate inputs for
    n : int, default = 1
        A hint for how many inputs are wanted. Usually n will be returned, but modules don't necessarily have to.
    device : mx.Device
        The device to do tracing on. If `None` (default), will use default device.
    dtype : mx.Dtype
        The dtype to trace with. If `None` (default), will use default dtype.

    Returns
    -------
    list of dict
        Tracing inputs in the format of dicts mapping method names like ``'forward'`` to tuples of
        arguments.
    """
    # Avoid circular imports
    from ._argtools import _get_device, _get_floating_dtype, _get_io_irreps, _rand_args, _to_device_dtype

    # - Get inputs -
    if hasattr(mod, _MAKE_TRACING_INPUTS):
        # This returns a trace_module style dict of method names to test inputs
        trace_inputs = mod._make_tracing_inputs(n)
        assert isinstance(trace_inputs, list)
        for d in trace_inputs:
            assert isinstance(d, dict), "_make_tracing_inputs must return a list of dict[str, tuple]"
            assert all(
                isinstance(k, str) and isinstance(v, tuple) for k, v in d.items()
            ), "_make_tracing_inputs must return a list of dict[str, tuple]"
    else:
        # Try to infer. This will throw if it can't.
        irreps_in, _ = _get_io_irreps(mod, irreps_out=[None])  # we're only trying to infer inputs
        trace_inputs = [{"forward": _rand_args(irreps_in)} for _ in range(n)]
    
    # - Put them on the right device -
    if device is None:
        device = _get_device(mod)
    if dtype is None:
        dtype = _get_floating_dtype(mod)
    
    # Move them
    trace_inputs = _to_device_dtype(trace_inputs, device, dtype)
    return trace_inputs


def trace_module(mod, inputs: dict = None, check_inputs: list = None, in_place: bool = True):
    """Trace a module.

    Parameters
    ----------
    mod : callable
        The module to trace
    inputs : dict
        Example inputs for tracing
    check_inputs : list of dict
        Additional inputs to check consistency
    in_place : bool, default True
        Whether to modify the module in-place

    Returns
    -------
    callable
        Traced module.
    """
    check_inputs = check_inputs or []

    # Set the compile mode for mod, temporarily
    old_mode = getattr(mod, _COMPILE_MODE_ATTR, None)
    if old_mode is not None and old_mode != "trace":
        warnings.warn(
            f"Trying to trace a module of type {type(mod).__name__} marked with @compile_mode != 'trace', expect errors!"
        )
    setattr(mod, _COMPILE_MODE_ATTR, "trace")

    # If inputs are provided, set make_tracing_input temporarily
    old_make_tracing_input = None
    if inputs is not None:
        old_make_tracing_input = getattr(mod, _MAKE_TRACING_INPUTS, None)
        setattr(mod, _MAKE_TRACING_INPUTS, lambda num: ([inputs] + check_inputs))

    # Compile
    out = compile(mod, in_place=in_place)

    # Restore old values, if we had them
    if old_mode is not None:
        setattr(mod, _COMPILE_MODE_ATTR, old_mode)
    if old_make_tracing_input is not None:
        setattr(mod, _MAKE_TRACING_INPUTS, old_make_tracing_input)
    return out


def trace(mod, example_inputs: tuple = None, check_inputs: list = None, in_place: bool = True):
    """Trace a module.

    Parameters
    ----------
    mod : callable
        The module to trace
    example_inputs : tuple
        Example inputs for tracing
    check_inputs : list of tuple
        Additional inputs to check consistency
    in_place : bool, default True
        Whether to modify the module in-place

    Returns
    -------
    callable
        Traced module.
    """
    check_inputs = check_inputs or []

    return trace_module(
        mod=mod,
        inputs=({"forward": example_inputs} if example_inputs is not None else None),
        check_inputs=[{"forward": c} for c in check_inputs],
        in_place=in_place,
    )


def script(mod, in_place: bool = True):
    """Script a module.

    Parameters
    ----------
    mod : callable
        The module to script
    in_place : bool, default True
        Whether to modify the module in-place

    Returns
    -------
    callable
        Scripted module.
    """
    # Set the compile mode for mod, temporarily
    old_mode = getattr(mod, _COMPILE_MODE_ATTR, None)
    if old_mode is not None and old_mode != "script":
        warnings.warn(
            f"Trying to script a module of type {type(mod).__name__} marked with @compile_mode != 'script', expect errors!"
        )
    setattr(mod, _COMPILE_MODE_ATTR, "script")

    # Compile
    out = compile(mod, in_place=in_place)

    # Restore old values, if we had them
    if old_mode is not None:
        setattr(mod, _COMPILE_MODE_ATTR, old_mode)

    return out


@contextmanager
def disable_mlx_codegen():
    """Context manager that disables MLX code generation optimizations."""
    # For MLX, this is a no-op since we don't have the same optimization controls
    # This is provided for API compatibility
    yield


def prepare(func: Callable, allow_autograd: bool = True) -> Callable:
    """Function transform that prepares a e3nn-mlx module for MLX compilation

    Args:
        func (Callable): A function that creates a callable module
        allow_autograd (bool, optional): Whether to allow autograd in compilation

    Returns:
        Callable: Decorated function that creates a MLX compile compatible module
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        with disable_mlx_codegen():
            model = func(*args, **kwargs)

        model = simplify(model)
        return model

    return wrapper


_SIMPLIFY_REGISTRY = set()


def simplify_if_compile(module):
    """Decorator to register a module for symbolic simplification

    The decorated module will be simplified using symbolic tracing.
    This constrains the module to not have any dynamic control flow.

    Args:
        module: the module to register

    Returns:
        the registered module
    """
    _SIMPLIFY_REGISTRY.add(module)
    return module


def simplify(module):
    """Recursively searches for registered modules to simplify with
    symbolic tracing to support compiling with MLX.

    Modules are registered with the `simplify_if_compile` decorator.

    Args:
        module: the module to simplify

    Returns:
        the simplified module
    """
    simplify_types = tuple(_SIMPLIFY_REGISTRY)

    for name, child in getattr(module, '__dict__', {}).items():
        if isinstance(child, simplify_types):
            # For MLX, we don't have direct equivalent of torch.fx.symbolic_trace
            # This is a no-op for now, but provides API compatibility
            pass
        else:
            simplify(child)

    return module