import mlx.core as mx

def complete_basis(vecs: mx.array, eps: float = 1e-9) -> mx.array:
    """
    Complete a set of vectors to a full orthonormal basis using the Gram-Schmidt process.

    Parameters
    ----------
    vecs : mx.array
        Tensor of shape (n, d) representing n vectors in d-dimensional space.
    eps : float, optional
        Epsilon for numerical stability.

    Returns
    -------
    mx.array
        Tensor of shape (m, d) where m is the number of new basis vectors added,
        such that the original vectors together with the returned vectors form a complete orthonormal basis.
    """
    assert vecs.ndim == 2
    dim = vecs.shape[1]

    # Normalize the input vectors
    base = [x / mx.linalg.norm(x) for x in vecs]

    expand = []
    for x in mx.eye(dim, dtype=vecs.dtype):
        # Gram-Schmidt orthogonalization
        for y in base + expand:
            x = x - mx.dot(x, y) * y
        norm_x = mx.linalg.norm(x)
        if norm_x > 2 * eps:
            x = x / norm_x
            x = mx.where(mx.abs(x) < eps, mx.zeros_like(x), x)
            # Ensure deterministic sign
            nonzero_idx = mx.nonzero(x)
            if nonzero_idx.shape[0] > 0:
                x = x * mx.sign(x[nonzero_idx[0, 0]])
            expand.append(x)

    if len(expand) > 0:
        expand = mx.stack(expand)
    else:
        expand = mx.zeros((0, dim), dtype=vecs.dtype)

    return expand
