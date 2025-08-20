"""
Safe linear algebra wrappers with CPU fallback for unsupported GPU ops.

These helpers attempt to run MX linalg ops; if the current device does not
support the operation, they fall back to CPU execution and return the result
as a regular array on the default device.
"""

from typing import Tuple
import mlx.core as mx


def _to_cpu(x: mx.array) -> mx.array:
    """Ensure `x` resides on CPU by constructing a CPU array from it."""
    return mx.array(x, device=mx.cpu)


def safe_inv(a: mx.array) -> mx.array:
    """Matrix inverse with automatic CPU fallback.

    Parameters
    - a: square matrix (..., n, n)
    """
    try:
        return mx.linalg.inv(a)
    except Exception:
        a_cpu = _to_cpu(a)
        res_cpu = mx.linalg.inv(a_cpu)
        return mx.array(res_cpu)


def safe_det(a: mx.array) -> mx.array:
    """Matrix determinant with automatic CPU fallback."""
    try:
        return mx.linalg.det(a)
    except Exception:
        a_cpu = _to_cpu(a)
        res_cpu = mx.linalg.det(a_cpu)
        return mx.array(res_cpu)


def safe_solve(a: mx.array, b: mx.array) -> mx.array:
    """Solve linear system Ax = b with automatic CPU fallback."""
    try:
        return mx.linalg.solve(a, b)
    except Exception:
        a_cpu = _to_cpu(a)
        b_cpu = _to_cpu(b)
        res_cpu = mx.linalg.solve(a_cpu, b_cpu)
        return mx.array(res_cpu)


def safe_pinv(a: mx.array, rcond: float = 1e-6) -> mx.array:
    """Pseudo-inverse using normal equations with CPU fallback.

    Note: MX may not expose SVD; this approximation uses (A^T A)^{-1} A^T.
    """
    try:
        at = mx.swapaxes(a, -2, -1)
        ata = mx.matmul(at, a)
        # Regularize
        eye = mx.eye(ata.shape[-1], dtype=ata.dtype)
        ata_reg = ata + (rcond * rcond) * eye
        ata_inv = safe_inv(ata_reg)
        return mx.matmul(ata_inv, at)
    except Exception:
        # CPU path for completeness
        return mx.array(safe_inv(mx.matmul(mx.swapaxes(_to_cpu(a), -2, -1), _to_cpu(a))) @ mx.swapaxes(_to_cpu(a), -2, -1))

