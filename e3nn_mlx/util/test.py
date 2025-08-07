import random
import math
import inspect
import itertools
import logging
from typing import Iterable, Optional, Union, List, Tuple, Any
import warnings

import numpy as np
import mlx.core as mx

from e3nn_mlx import o3
from e3nn_mlx.o3 import _irreps
from e3nn_mlx.util import broadcast_shapes

# pylint: disable=unused-variable

# Make a logger for reporting error statistics
logger = logging.getLogger(__name__)


def _logging_name(func) -> str:
    """Get a decent string representation of ``func`` for logging"""
    if inspect.isfunction(func):
        return func.__name__
    else:
        return repr(func)


# The default float tolerance
FLOAT_TOLERANCE = {mx.float32: 1e-3, mx.float64: 1e-9}


def random_irreps(
    n: int = 1,
    lmax: int = 4,
    mul_min: int = 0,
    mul_max: int = 5,
    len_min: int = 0,
    len_max: int = 4,
    clean: bool = False,
    allow_empty: bool = True,
):
    r"""Generate random irreps parameters for testing.

    Parameters
    ----------
        n : int, optional
            How many to generate; defaults to 1.
        lmax : int, optional
            The maximum L to generate (inclusive); defaults to 4.
        mul_min : int, optional
            The smallest multiplicity to generate, defaults to 0.
        mul_max : int, optional
            The largest multiplicity to generate, defaults to 5.
        len_min : int, optional
            The smallest number of irreps to generate, defaults to 0.
        len_max : int, optional
            The largest number of irreps to generate, defaults to 4.
        clean : bool, optional
            If ``True``, only ``o3.Irreps`` objects will be returned. If ``False`` (the default),
            ``e3nn.o3.Irreps``-like objects like strings and lists of tuples can be returned.
        allow_empty : bool, optional
            Whether to allow generating empty ``e3nn.o3.Irreps``.
    Returns
    -------
        An irreps-like object if ``n == 1`` or a list of them if ``n > 1``
    """
    assert n >= 1
    assert lmax >= 0
    assert mul_min >= 0
    assert mul_max >= mul_min

    if not allow_empty and len_min == 0:
        len_min = 1
    assert len_min >= 0
    assert len_max >= len_min

    out = []
    for _ in range(n):
        this_irreps = []
        for _ in range(random.randint(len_min, len_max)):
            this_irreps.append((random.randint(mul_min, mul_max), (random.randint(0, lmax), random.choice((1, -1)))))
        if not allow_empty and all(m == 0 for m, _ in this_irreps):
            this_irreps[-1] = (random.randint(1, mul_max), this_irreps[-1][1])
        this_irreps = o3.Irreps(this_irreps)

        if clean:
            outtype = "irreps"
        else:
            outtype = random.choice(("irreps", "str", "list"))
        if outtype == "irreps":
            out.append(this_irreps)
        elif outtype == "str":
            out.append(repr(this_irreps))
        elif outtype == "list":
            out.append([(mul_ir.mul, (mul_ir.ir.l, mul_ir.ir.p)) for mul_ir in this_irreps])

    if n == 1:
        return out[0]
    else:
        return out


def format_equivariance_error(errors: dict) -> str:
    """Format the dictionary returned by ``equivariance_error`` into a readable string.

    Parameters
    ----------
        errors : dict
            A dictionary of errors returned by ``equivariance_error``.

    Returns
    -------
        A string.
    """
    return "\n".join(
        "(parity_k={:d}, did_translate={}) -> max error={:.3e} in argument {}".format(
            int(k[0]), bool(k[1]), float(v.max()), int(v.argmax())
        )
        for k, v in errors.items()
    )


def assert_equivariant(func, args_in=None, irreps_in=None, irreps_out=None, tolerance=None, **kwargs) -> dict:
    r"""Assert that ``func`` is equivariant.

    Parameters
    ----------
        func : callable
            the function to test
        args_in : list or None
            the original input arguments for the function. If ``None`` and the function has ``irreps_in``
            consisting only of ``o3.Irreps`` and ``'cartesian'``, random test inputs will be generated.
        irreps_in : object
            see ``equivariance_error``
        irreps_out : object
            see ``equivariance_error``
        tolerance : float or None
            the threshold below which the equivariance error must fall.
            If ``None``, (the default), ``FLOAT_TOLERANCE[mx.float32]`` is used.
        **kwargs : kwargs
            passed through to ``equivariance_error``.

    Returns
    -------
    The same as ``equivariance_error``: a dictionary mapping tuples ``(parity_k, did_translate)`` to errors
    """
    # Prevent pytest from showing this function in the traceback
    __tracebackhide__ = True

    args_in, irreps_in, irreps_out = _get_args_in(func, args_in=args_in, irreps_in=irreps_in, irreps_out=irreps_out)

    # Get error
    errors = equivariance_error(func, args_in=args_in, irreps_in=irreps_in, irreps_out=irreps_out, **kwargs)

    logger.info(
        "Tested equivariance of `%s` -- max componentwise errors:\n%s",
        _logging_name(func),
        format_equivariance_error(errors),
    )

    # Check it
    if tolerance is None:
        tolerance = FLOAT_TOLERANCE[mx.float32]

    problems = {case: err for case, err in errors.items() if err.max() > tolerance}

    if len(problems) != 0:
        errstr = "Largest componentwise equivariance error was too large for: "
        errstr += format_equivariance_error(problems)
        assert len(problems) == 0, errstr

    return errors


def equivariance_error(
    func,
    args_in,
    irreps_in=None,
    irreps_out=None,
    ntrials: int = 1,
    do_parity: bool = True,
    do_translation: bool = True,
    transform_dtype=mx.float32,
):
    r"""Get the maximum equivariance error for ``func`` over ``ntrials``

    Each trial randomizes the equivariant transformation tested.

    Parameters
    ----------
    func : callable
        the function to test
    args_in : list
        the original inputs to pass to ``func``.
    irreps_in : list of `e3nn.o3.Irreps` or `e3nn.o3.Irreps`
        the input irreps for each of the arguments in ``args_in``. If left as the default of ``None``,
        ``get_io_irreps`` will be used to try to infer them. If a sequence is provided, valid elements
        are also the string ``'cartesian'``, which denotes that the corresponding input should be dealt
        with as cartesian points in 3D, and ``None``, which indicates that the argument should not be transformed.
    irreps_out : list of `e3nn.o3.Irreps` or `e3nn.o3.Irreps`
        the out irreps for each of the return values of ``func``. Accepts similar values to ``irreps_in``.
    ntrials : int
        run this many trials with random transforms
    do_parity : bool
        whether to test parity
    do_translation : bool
        whether to test translation for ``'cartesian'`` inputs

    Returns
    -------
    dictionary mapping tuples ``(parity_k, did_translate)`` to an array of errors,
    each entry the biggest over all trials for that output, in order.
    """
    irreps_in, irreps_out = _get_io_irreps(func, irreps_in=irreps_in, irreps_out=irreps_out)

    if do_parity:
        parity_ks = [0, 1]
    else:
        parity_ks = [0]

    if "cartesian_points" not in irreps_in:
        # There's nothing to translate
        do_translation = False
    if do_translation:
        do_translation = [False, True]
    else:
        do_translation = [False]

    tests = list(itertools.product(parity_ks, do_translation))

    neg_inf = -float("Inf")
    biggest_errs = {test: mx.full((len(irreps_out),), neg_inf, dtype=transform_dtype) for test in tests}

    for trial in range(ntrials):
        for this_test in tests:
            parity_k, this_do_translate = this_test
            # Build a rotation matrix for point data
            rot_mat = o3.rand_matrix(dtype=transform_dtype)
            # add parity
            rot_mat *= (-1) ** parity_k
            # build translation
            translation = 10 * mx.random.normal((1, 3), dtype=rot_mat.dtype) if this_do_translate else 0.0

            # Evaluate the function on rotated arguments:
            rot_args = _transform(args_in, irreps_in, rot_mat, translation)
            x1 = func(*rot_args)
            if isinstance(x1, mx.array):
                x1 = [x1]
            elif isinstance(x1, (list, tuple)):
                x1 = list(x1)
            else:
                raise TypeError(f"equivariance_error cannot handle output type {type(x1)}")
            # convert into the transform dtype for computing the difference
            x1 = [mx.array(t, dtype=transform_dtype) for t in x1]

            # Evaluate the function on the arguments, then apply group action:
            x2 = func(*args_in)
            if isinstance(x2, mx.array):
                x2 = [x2]
            elif isinstance(x2, (list, tuple)):
                x2 = list(x2)
            else:
                raise TypeError(f"equivariance_error cannot handle output type {type(x2)}")
            x2 = [mx.array(t) for t in x2]

            # confirm sanity
            assert len(x1) == len(x2)
            # Handle tensor products that return single concatenated output
            if hasattr(func, 'irreps_out') and isinstance(func.irreps_out, o3.Irreps):
                # For tensor products, expect single output
                assert len(x1) == 1, f"Tensor product returned {len(x1)} outputs, expected 1"
            else:
                # For other functions, expect one output per irrep
                assert len(x1) == len(irreps_out), f"Function returned {len(x1)} outputs, but irreps_out has {len(irreps_out)} elements: {irreps_out}"

            # apply the group action to x2
            # For tensor products, we need to handle irreps_out as a single Irreps object
            if hasattr(func, 'irreps_out') and isinstance(func.irreps_out, o3.Irreps):
                # For tensor products, use the single Irreps object directly
                transform_irreps_out = [func.irreps_out]
            else:
                transform_irreps_out = irreps_out
            x2 = _transform(x2, transform_irreps_out, rot_mat, translation, output_transform_dtype=True)

            # compute errors in the transform dtype
            errors = mx.stack([mx.abs(a - b).max() for a, b in zip(x1, x2)])

            biggest_errs[this_test] = mx.where(errors > biggest_errs[this_test], errors, biggest_errs[this_test])

    # convert errors back to default dtype to return:
    return {k: mx.array(v, dtype=mx.float32) for k, v in biggest_errs.items()}


def _get_io_irreps(func, irreps_in=None, irreps_out=None):
    """Get the input and output irreps for a function."""
    if irreps_in is None:
        # Try to infer from function signature
        if hasattr(func, 'irreps_in1') and hasattr(func, 'irreps_in2'):
            # TensorProduct case
            irreps_in = [func.irreps_in1, func.irreps_in2]
        elif hasattr(func, 'irreps_in') and hasattr(func, 'irreps_out'):
            # Linear case
            irreps_in = [func.irreps_in]
        else:
            raise ValueError("Cannot infer irreps_in from function signature")
    
    if irreps_out is None:
        if hasattr(func, 'irreps_out'):
            irreps_out = func.irreps_out
            # For tensor products, keep as single Irreps object
            # The test framework will handle splitting it into individual irreps
        else:
            raise ValueError("Cannot infer irreps_out from function signature")
    else:
        # Ensure irreps_out is a list even when provided explicitly
        if not isinstance(irreps_out, (list, tuple)):
            irreps_out = [irreps_out]
    
    return irreps_in, irreps_out


def _get_args_in(func, args_in=None, irreps_in=None, irreps_out=None):
    """Get arguments for testing."""
    if args_in is None:
        irreps_in, _ = _get_io_irreps(func, irreps_in=irreps_in, irreps_out=irreps_out)
        args_in = _rand_args(irreps_in)
    else:
        irreps_in, _ = _get_io_irreps(func, irreps_in=irreps_in, irreps_out=irreps_out)
    
    return args_in, irreps_in, irreps_out


def _rand_args(irreps_in, batch_size: int = 1):
    """Generate random arguments for testing."""
    args = []
    for irreps in irreps_in:
        if irreps == 'cartesian_points':
            # Generate random 3D points
            args.append(mx.random.normal((batch_size, 3)))
        elif isinstance(irreps, o3.Irreps):
            # Generate random irreps data
            args.append(mx.random.normal((batch_size, irreps.dim)))
        elif irreps is None:
            # Skip this argument
            continue
        else:
            raise ValueError(f"Unsupported irreps type: {type(irreps)}")
    return args


def _transform(args, irreps, rot_mat, translation=0.0, output_transform_dtype=False):
    """Apply transformation to arguments based on irreps."""
    transformed = []
    dtype = rot_mat.dtype if output_transform_dtype else mx.float32
    
    for arg, irreps_type in zip(args, irreps):
        if irreps_type == 'cartesian_points':
            # Transform points
            transformed.append(mx.matmul(arg, rot_mat.T) + translation)
        elif isinstance(irreps_type, o3.Irreps):
            # Transform irreps data by applying rotation to each irrep
            transformed_arg = mx.array(arg, dtype=dtype)
            slices = []
            start = 0
            for mul_ir in irreps_type:
                end = start + mul_ir.dim
                slice_data = transformed_arg[:, start:end]
                if mul_ir.ir.l > 0:  # Only transform non-scalar irreps
                    # Get Wigner D matrix for this irrep
                    angles = o3.matrix_to_angles(rot_mat)
                    D = o3.wigner_D(mul_ir.ir.l, *angles)
                    # Reshape the slice and apply transformation
                    slice_data = slice_data.reshape(-1, mul_ir.mul, mul_ir.ir.dim)
                    
                    # Handle batched D matrices - apply each D matrix to corresponding batch
                    if D.ndim == 3:  # Batch of D matrices
                        transformed_slices = []
                        for i in range(D.shape[0]):
                            # Apply transformation for this batch element
                            transformed_slice = mx.einsum('ij,uj->ui', D[i], slice_data[i])
                            transformed_slices.append(transformed_slice)
                        slice_data = mx.stack(transformed_slices)
                    else:  # Single D matrix
                        slice_data = mx.einsum('ij,zuj->zui', D, slice_data)
                    
                    slice_data = slice_data.reshape(-1, mul_ir.dim)
                slices.append(slice_data)
                start = end
            # Concatenate all slices
            transformed.append(mx.concatenate(slices, axis=1))
        elif isinstance(irreps_type, _irreps._MulIr):
            # Single MulIr case
            transformed.append(mx.array(arg, dtype=dtype))
        elif irreps_type is None:
            # No transformation
            transformed.append(arg)
        else:
            raise ValueError(f"Unsupported irreps type: {type(irreps_type)}")
    
    return transformed


def set_random_seeds() -> None:
    """Set the random seeds to try to get some reproducibility"""
    mx.random.seed(0)
    random.seed(0)
    np.random.seed(0)