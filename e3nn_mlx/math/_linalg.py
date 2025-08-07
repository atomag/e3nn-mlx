from typing import Tuple
from e3nn_mlx.util._array_workarounds import array_at_set_workaround, spherical_harmonics_set_workaround


import mlx.core as mx


def direct_sum(*matrices):
    r"""Direct sum of matrices, put them in the diagonal"""
    front_indices = matrices[0].shape[:-2]
    m = sum(x.shape[-2] for x in matrices)
    n = sum(x.shape[-1] for x in matrices)
    total_shape = list(front_indices) + [m, n]
    out = mx.zeros(total_shape, dtype=matrices[0].dtype)
    i, j = 0, 0
    for x in matrices:
        m, n = x.shape[-2:]
        out[..., i : i + m, j : j + n] = x
        i += m
        j += n
    return out


def orthonormalize(original: mx.array, eps: float = 1e-9) -> Tuple[mx.array, mx.array]:
    r"""orthonomalize vectors

    Parameters
    ----------
    original : `mx.array`
        list of the original vectors :math:`x`

    eps : float
        a small number

    Returns
    -------
    final : `mx.array`
        list of orthonomalized vectors :math:`y`

    matrix : `mx.array`
        the matrix :math:`A` such that :math:`y = A x`
    """
    assert original.ndim == 2
    dim = original.shape[1]

    final = []
    matrix = []

    for i, x in enumerate(original):
        # x = sum_i cx_i original_i
        cx = mx.zeros(len(original), dtype=original.dtype)
        cx = array_at_set_workaround(cx, i, 1)
        for j, y in enumerate(final):
            c = mx.dot(x, y)
            x = x - c * y
            cx = cx - c * matrix[j]
        if mx.linalg.norm(x) > 2 * eps:
            c = 1 / mx.linalg.norm(x)
            x = c * x
            cx = c * cx
            x = mx.where(mx.abs(x) < eps, 0, x)
            cx = mx.where(mx.abs(cx) < eps, 0, cx)
            c = mx.sign(x[mx.nonzero(x)[0, 0]])
            x = c * x
            cx = c * cx
            final.append(x)
            matrix.append(cx)

    final = mx.stack(final) if len(final) > 0 else mx.zeros((0, dim), dtype=original.dtype)
    matrix = mx.stack(matrix) if len(matrix) > 0 else mx.zeros((0, len(original)), dtype=original.dtype)

    return final, matrix


def complete_basis(vecs: mx.array, eps: float = 1e-9) -> mx.array:
    assert vecs.ndim == 2
    dim = vecs.shape[1]

    base = [x / mx.linalg.norm(x) for x in vecs]

    expand = []
    for x in mx.eye(dim, dtype=vecs.dtype):
        for y in base + expand:
            x = x - mx.dot(x, y) * y
        if mx.linalg.norm(x) > 2 * eps:
            x = x / mx.linalg.norm(x)
            x = mx.where(mx.abs(x) < eps, mx.zeros_like(x), x)
            x = x * mx.sign(x[mx.nonzero(x)[0, 0]])
            expand.append(x)

    expand = mx.stack(expand) if len(expand) > 0 else mx.zeros((0, dim), dtype=vecs.dtype)

    return expand