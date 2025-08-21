"""
Wigner D-matrices and 3j symbols for O(3) group in MLX.
Mathematically correct implementation based on Racah formula.
"""

import mlx.core as mx
import math
from math import factorial, sqrt
from typing import Tuple
from functools import lru_cache
from ..util._array_workarounds import array_at_set_workaround


def _racah_factorial(n: int) -> float:
    """Compute factorial with handling for negative numbers (returns 1)."""
    if n < 0:
        return 1.0
    return float(factorial(n))


def _su2_clebsch_gordan_coeff(j1: float, m1: float, j2: float, m2: float, j3: float, m3: float) -> float:
    """
    Compute SU(2) Clebsch-Gordan coefficient using Racah formula.
    
    Parameters
    ----------
    j1, m1 : float
        Angular momentum and projection for first particle
    j2, m2 : float  
        Angular momentum and projection for second particle
    j3, m3 : float
        Angular momentum and projection for coupled system
        
    Returns
    -------
    coeff : float
        Clebsch-Gordan coefficient
    """
    # Check conservation of angular momentum projection
    if abs(m3 - (m1 + m2)) > 1e-10:
        return 0.0
    
    # Check triangle condition
    if abs(j1 - j2) > j3 or j1 + j2 < j3:
        return 0.0
    
    # Racah formula implementation
    def f(n):
        return _racah_factorial(n)
    
    # Precompute common factors
    C = (
        (2.0 * j3 + 1.0)
        * f(j3 + j1 - j2) * f(j3 - j1 + j2) * f(j1 + j2 - j3)
        * f(j3 + m3) * f(j3 - m3)
        / (f(j1 + j2 + j3 + 1) * f(j1 - m1) * f(j1 + m1) * f(j2 - m2) * f(j2 + m2))
    ) ** 0.5
    
    # Sum over auxiliary index v
    S = 0.0
    vmin = max(0, j2 - j1 - m3)
    vmax = min(j2 + j3 - m1, j3 + j1 - j2)
    
    for v in range(int(vmin), int(vmax) + 1):
        numerator = (
            f(j2 + j3 + m1 - v) * f(j1 - m1 + v)
        )
        denominator = (
            f(v) * f(j3 - j1 + j2 - v) * f(j3 + m3 - v) * f(v + j1 - j2 - m3)
        )
        
        phase = (-1.0) ** int(v + j2 + m2)
        S += phase * numerator / denominator
    
    return C * S


def _change_basis_real_to_complex(l: int) -> mx.array:
    """Change of basis matrix from real to complex spherical harmonics.

    Matches e3nn (PyTorch) conventions, including the (-1j)^l phase to make
    CG coefficients real.
    """
    dim = 2 * l + 1
    Q = mx.zeros((dim, dim), dtype=mx.complex64)

    # m < 0 rows
    for m in range(-l, 0):
        Q = array_at_set_workaround(Q, (l + m, l + abs(m)), 1.0 / sqrt(2.0))
        Q = array_at_set_workaround(Q, (l + m, l - abs(m)), mx.array(-1j / sqrt(2.0), dtype=mx.complex64))

    # m == 0 row
    Q = array_at_set_workaround(Q, (l, l), 1.0)

    # m > 0 rows
    for m in range(1, l + 1):
        s = (-1.0) ** m
        Q = array_at_set_workaround(Q, (l + m, l + m), s / sqrt(2.0))
        Q = array_at_set_workaround(Q, (l + m, l - m), mx.array(1j * s / sqrt(2.0), dtype=mx.complex64))

    # Global phase (-1j)^l
    phase = (-1j) ** l
    Q = Q * phase
    return Q


def _so3_clebsch_gordan(l1: int, l2: int, l3: int) -> mx.array:
    """
    Compute SO(3) Clebsch-Gordan coefficients from SU(2) coefficients.
    
    Parameters
    ----------
    l1, l2, l3 : int
        Angular momentum quantum numbers
        
    Returns
    -------
    C : mx.array
        SO(3) Clebsch-Gordan coefficients of shape (2l1+1, 2l2+1, 2l3+1)
    """
    # Check triangle condition
    if abs(l1 - l2) > l3 or l1 + l2 < l3:
        return mx.zeros((2*l1+1, 2*l2+1, 2*l3+1))
    
    dim1, dim2, dim3 = 2*l1+1, 2*l2+1, 2*l3+1
    
    # Compute basis transformation matrices
    Q1 = _change_basis_real_to_complex(l1)
    Q2 = _change_basis_real_to_complex(l2)
    Q3 = _change_basis_real_to_complex(l3)
    
    # Compute SU(2) coefficients
    C_su2 = mx.zeros((dim1, dim2, dim3), dtype=mx.complex64)
    
    for m1 in range(-l1, l1 + 1):
        for m2 in range(-l2, l2 + 1):
            for m3 in range(-l3, l3 + 1):
                idx1, idx2, idx3 = m1 + l1, m2 + l2, m3 + l3
                coeff = _su2_clebsch_gordan_coeff(l1, m1, l2, m2, l3, m3)
                C_su2 = array_at_set_workaround(C_su2, (idx1, idx2, idx3), coeff)
    
    # Transform to SO(3) basis
    # C_SO3 = Q1^T @ C_SU2 @ Q2 @ Q3^*
    C_so3 = mx.zeros((dim1, dim2, dim3), dtype=mx.complex64)
    
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                # Perform the basis transformation
                temp = 0.0 + 0.0j
                for i1 in range(dim1):
                    for j1 in range(dim2):
                        for k1 in range(dim3):
                            temp += (
                                Q1[i1, i] * Q2[j1, j] * mx.conj(Q3[k1, k]) * 
                                C_su2[i1, j1, k1]
                            )
                C_so3 = array_at_set_workaround(C_so3, (i, j, k), temp)
    
    # Ensure real and normalize
    C_real = mx.real(C_so3)
    C_norm = mx.linalg.norm(C_real)
    
    if C_norm > 1e-10:
        C_real = C_real / C_norm
    
    return C_real


@lru_cache(maxsize=None)
def wigner_3j(l1: int, l2: int, l3: int) -> mx.array:
    """
    Compute Wigner 3j symbols for given angular momenta.
    
    Mathematically correct implementation using Racah formula.
    
    Parameters
    ----------
    l1, l2, l3 : int
        Angular momentum quantum numbers
        
    Returns
    -------
    w3j : mx.array
        Wigner 3j symbols of shape (2l1+1, 2l2+1, 2l3+1)
    """
    # Check triangle condition
    if abs(l1 - l2) > l3 or l1 + l2 < l3:
        return mx.zeros((2*l1+1, 2*l2+1, 2*l3+1))
    
    # Wigner 3j symbols are related to Clebsch-Gordan coefficients by:
    # ( l1  l2  l3 ) = (-1)^(l1-l2-m3) / sqrt(2*l3+1) * <l1 m1 l2 m2 | l3 -m3>
    # ( m1  m2  m3 )
    
    cg = _so3_clebsch_gordan(l1, l2, l3)
    w3j = mx.zeros_like(cg)
    
    for m1 in range(-l1, l1 + 1):
        for m2 in range(-l2, l2 + 1):
            for m3 in range(-l3, l3 + 1):
                idx1, idx2, idx3 = m1 + l1, m2 + l2, m3 + l3
                
                # Apply phase factor and normalization
                phase = (-1.0) ** int(l1 - l2 - m3)
                norm = 1.0 / sqrt(2.0 * l3 + 1.0)
                
                w3j = array_at_set_workaround(w3j, (idx1, idx2, idx3), phase * norm * cg[idx1, idx2, idx3])
    
    return w3j


def wigner_D(l: int, alpha: mx.array, beta: mx.array, gamma: mx.array) -> mx.array:
    """Wigner D matrix in real basis using ZYZ-parameter closed form.

    Converts Y-X-Y angles to Z-Y-Z, then applies D_complex = e^{-i m α'} d(β') e^{-i m' γ'} and
    transforms to real basis.
    """
    from ._rotation import angles_to_matrix

    dim = 2 * l + 1

    def factorial(n: int) -> float:
        from math import factorial as f
        if n < 0:
            return 1.0
        return float(f(n))

    def small_d(l: int, beta: mx.array) -> mx.array:
        is_scalar = (beta.ndim == 0)
        beta_in = beta[None] if is_scalar else beta
        B = beta_in.shape[0]
        cb = mx.cos(0.5 * beta_in)
        sb = mx.sin(0.5 * beta_in)
        d = mx.zeros((B, dim, dim), dtype=beta_in.dtype)
        fact_cache = {}
        def F(n):
            if n not in fact_cache:
                fact_cache[n] = factorial(n)
            return fact_cache[n]
        for m in range(-l, l + 1):
            for mp in range(-l, l + 1):
                pref = (F(l + m) * F(l - m) * F(l + mp) * F(l - mp)) ** 0.5
                kmin = max(0, m - mp)
                kmax = min(l + m, l - mp)
                val = mx.zeros((B,), dtype=beta_in.dtype)
                for k in range(int(kmin), int(kmax) + 1):
                    denom = F(l + m - k) * F(l - mp - k) * F(k) * F(k + mp - m)
                    if denom == 0:
                        continue
                    pcos = 2 * l + m - mp - 2 * k
                    psin = 2 * k + mp - m
                    term = ((-1.0) ** (k - m + mp)) * (pref / denom)
                    term = term * (mx.power(cb, pcos) * mx.power(sb, psin))
                    val = val + term
                for b in range(B):
                    d = array_at_set_workaround(d, (b, m + l, mp + l), val[b])
        return d[0] if is_scalar else d

    def to_batched(x):
        return x[None] if x.ndim == 0 else x

    alpha_b = to_batched(alpha)
    beta_b = to_batched(beta)
    gamma_b = to_batched(gamma)
    B = max(alpha_b.shape[0], beta_b.shape[0], gamma_b.shape[0])
    if alpha_b.shape[0] != B:
        alpha_b = mx.broadcast_to(alpha_b, (B,))
    if beta_b.shape[0] != B:
        beta_b = mx.broadcast_to(beta_b, (B,))
    if gamma_b.shape[0] != B:
        gamma_b = mx.broadcast_to(gamma_b, (B,))

    # Convert Y-X-Y to Z-Y-Z via rotation matrix extraction
    R = angles_to_matrix(alpha_b, beta_b, gamma_b)
    # Extract ZYZ Euler angles α', β', γ'
    # beta' = arccos(R[2,2])
    beta_p = mx.arccos(mx.clip(R[:, 2, 2] if R.ndim == 3 else R[2, 2], -1.0, 1.0))
    # alpha' = atan2(R[1,2], R[0,2])
    if R.ndim == 3:
        alpha_p = mx.arctan2(R[:, 1, 2], R[:, 0, 2])
        gamma_p = mx.arctan2(R[:, 2, 1], -R[:, 2, 0])
    else:
        alpha_p = mx.arctan2(R[1, 2], R[0, 2])
        gamma_p = mx.arctan2(R[2, 1], -R[2, 0])

    # Build D_complex from ZYZ angles
    m_vals = mx.arange(-l, l + 1, dtype=beta_p.dtype)
    Ealpha = mx.exp((-1j) * alpha_p[:, None] * m_vals[None, :]) if alpha_p.ndim == 1 else mx.exp((-1j) * alpha_p * m_vals)
    Egamma = mx.exp((-1j) * gamma_p[:, None] * m_vals[None, :]) if gamma_p.ndim == 1 else mx.exp((-1j) * gamma_p * m_vals)
    dmat = small_d(l, beta_p)
    if dmat.ndim == 2:
        dmat = dmat[None]
    D_complex = (Ealpha[:, :, None].astype(mx.complex64) * dmat.astype(mx.complex64)) * Egamma[:, None, :].astype(mx.complex64)

    Q = _change_basis_real_to_complex(l)
    Qh = mx.conj(Q).T
    D_left = mx.einsum("ij,bjk->bik", Qh, D_complex)
    D_real_c = mx.einsum("bij,jk->bik", D_left, Q)
    # Enforce real-valued representation
    D_real = mx.real(D_real_c)

    return D_real if (alpha.ndim > 0 or beta.ndim > 0 or gamma.ndim > 0) else D_real[0]


def change_basis_real_to_complex(l: int) -> mx.array:
    """
    Compute basis transformation matrix from real to complex spherical harmonics.
    
    This function is exposed for external use and provides a public interface
    to the internal _change_basis_real_to_complex function.
    
    Parameters
    ----------
    l : int
        Angular momentum quantum number
        
    Returns
    -------
    Q : mx.array
        Transformation matrix of shape (2l+1, 2l+1)
    """
    return _change_basis_real_to_complex(l)


def su2_generators(j: int) -> mx.array:
    """
    Compute SU(2) generators for angular momentum j.
    
    Parameters
    ----------
    j : int
        Angular momentum quantum number
        
    Returns
    -------
    generators : mx.array
        SU(2) generators. Jx is real, Jy and Jz are complex with real and imaginary parts
        stored as separate arrays. Shape is (5, 2j+1, 2j+1) where:
        - generators[0] = Jx (real)
        - generators[1] = Re(Jy), generators[2] = Im(Jy) 
        - generators[3] = Re(Jz), generators[4] = Im(Jz)
    """
    m = mx.arange(-j, j + 1)
    dim = 2*j + 1
    
    # Raising operator
    raising = mx.zeros((dim, dim))
    for i in range(2*j):
        coeff = -mx.sqrt(j * (j + 1) - m[i] * (m[i] + 1))
        raising = raising.at[i + 1, i].add(coeff)
    
    # Lowering operator
    lowering = mx.zeros((dim, dim))
    for i in range(1, 2*j + 1):
        coeff = mx.sqrt(j * (j + 1) - m[i] * (m[i] - 1))
        lowering = lowering.at[i - 1, i].add(coeff)
    
    # Construct generators - handle complex parts separately
    Jx = 0.5 * (raising + lowering)
    
    # Jy has imaginary components, return as real and imaginary parts
    Jy_real = mx.zeros((dim, dim))
    Jy_imag = -0.5 * (raising - lowering)
    
    # Jz has imaginary components on diagonal
    Jz_real = mx.zeros((dim, dim))
    Jz_imag = mx.diag(m)
    
    # Stack all generators as real arrays
    # For complex generators, we stack real and imaginary parts
    generators = mx.stack([
        Jx,  # Real
        Jy_real, Jy_imag,  # Complex Jy as real + imag
        Jz_real, Jz_imag   # Complex Jz as real + imag
    ])
    
    return generators


def so3_generators(l: int) -> mx.array:
    """Compute SO(3) generators (Lx, Ly, Lz) numerically from wigner_D at identity.

    Ensures consistency with wigner_D implementation and Y-X-Y convention.
    """
    eps = mx.array(1e-3)
    I = wigner_D(l, mx.array(0.0), mx.array(0.0), mx.array(0.0))
    # Lx = d/dβ at 0
    Dp = wigner_D(l, mx.array(0.0), eps, mx.array(0.0))
    Dm = wigner_D(l, mx.array(0.0), -eps, mx.array(0.0))
    Lx = (Dp - Dm) / (2 * float(eps.tolist()))
    # Ly = d/dα at 0
    Dp = wigner_D(l, eps, mx.array(0.0), mx.array(0.0))
    Dm = wigner_D(l, -eps, mx.array(0.0), mx.array(0.0))
    Ly = (Dp - Dm) / (2 * float(eps.tolist()))
    # Lz = [Lx, Ly]
    Lz = mx.matmul(Ly, Lx) - mx.matmul(Lx, Ly)
    return mx.stack([Lx, Ly, Lz])


def clebsch_gordan(l1: int, l2: int, l3: int) -> mx.array:
    """
    Compute Clebsch-Gordan coefficients.
    
    Returns a 3D array of shape (2*l1+1, 2*l2+1, 2*l3+1)
    """
    # This is related to the Wigner 3j symbols by a phase factor
    w3j = wigner_3j(l1, l2, l3)
    # The exact relation involves a phase factor: 
    # <l1 m1 l2 m2 | l3 m3> = (-1)^(l1-l2+m3) * sqrt(2*l3+1) * wigner_3j(l1, l2, l3; m1, m2, -m3)
    return w3j * mx.sqrt(2*l3 + 1)
