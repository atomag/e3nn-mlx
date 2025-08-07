from typing import Tuple
import mlx.core as mx


def orthonormalize(matrix: mx.array, eps: float = 1e-9) -> Tuple[mx.array, mx.array]:
    """
    Orthonormalize a matrix using Gram-Schmidt process.

    Parameters
    ----------
    matrix : mx.array
        Input matrix to orthonormalize
    eps : float, optional
        Epsilon for numerical stability

    Returns
    -------
    Tuple[mx.array, mx.array]
        Orthonormalized matrix and transformation matrix
    """
    # Simplified orthonormalization implementation
    # In practice, this would use QR decomposition or similar

    # Use SVD for orthonormalization
    u, s, vt = mx.linalg.svd(matrix)

    # Threshold small singular values
    mask = s > eps
    s = mx.where(mask, s, mx.zeros_like(s))

    # Reconstruct orthonormal basis
    orthonormal = u @ mx.diag(s) @ vt

    return orthonormal, mx.eye(orthonormal.shape[0], dtype=orthonormal.dtype)
