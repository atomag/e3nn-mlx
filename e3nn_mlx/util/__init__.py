"""Utility functions for e3nn-mlx."""

from typing import Iterable, Any

# Import validation utilities
from ._validation import (
    validate_type,
    validate_positive_number,
    validate_range,
    validate_array_like,
    validate_array_dimensions,
    validate_array_shape,
    validate_string_choice,
    validate_list_length,
    validate_boolean,
    validate_not_none,
    ValidationError,
    InputValidationError,
    ShapeValidationError,
    TypeValidationError,
    ValueValidationError,
)

# Import JIT compilation utilities
from .jit import (
    compile_mode,
    get_compile_mode,
    compile,
    trace_module,
    trace,
    script,
    disable_mlx_codegen,
    prepare,
    simplify_if_compile,
    simplify,
)

# Import argument tools
from ._argtools import (
    _transform,
    _get_io_irreps,
    _get_args_in,
    _rand_args,
    _get_device,
    _get_floating_dtype,
    _to_device_dtype,
    _make_tracing_inputs,
)

# Import context management
from ._context import (
    default_dtype,
    default_device,
    get_default_dtype,
    get_default_device,
    set_default_dtype,
    set_default_device,
    explicit_default_types,
    resolve_dtype,
    resolve_device,
)

# Import data types
from .datatypes import (
    Chunk,
    Path,
    TensorProductPath,
    OptimizedOperation,
    Instruction,
    chunk_from_slice,
    path_from_instructions,
    validate_chunk,
    validate_path,
)


def prod(iterable: Iterable[Any]) -> int:
    """Compute the product of elements in an iterable."""
    result = 1
    for x in iterable:
        result *= x
    return result


def broadcast_shapes(*shapes):
    """Broadcast shapes like numpy."""
    if not shapes:
        return ()
    
    max_len = max(len(shape) for shape in shapes)
    
    # Pad shapes with 1s on the left
    padded_shapes = []
    for shape in shapes:
        if len(shape) < max_len:
            padded = (1,) * (max_len - len(shape)) + shape
        else:
            padded = shape
        padded_shapes.append(padded)
    
    # Check broadcasting compatibility
    result = []
    for dims in zip(*padded_shapes):
        max_dim = 1
        for d in dims:
            if d != 1 and max_dim != 1 and d != max_dim:
                raise ValueError(f"Broadcasting failed: shapes {shapes} are incompatible")
            max_dim = max(max_dim, d)
        result.append(max_dim)
    
    return tuple(result)

# Safe linalg fallbacks
from .safe_linalg import (
    safe_inv,
    safe_det,
    safe_solve,
    safe_pinv,
)
