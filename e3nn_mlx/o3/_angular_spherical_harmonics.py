r"""Spherical Harmonics as functions of Euler angles"""
import math
from typing import List, Tuple

import mlx.core as mx
from sympy import Integer, Poly, diff, factorial, pi, sqrt, symbols


class SphericalHarmonicsAlphaBeta:
    """MLX version of spherical harmonics alpha beta.

    Parameters are identical to the original PyTorch version.
    """

    def __init__(self, l, normalization: str = "integral") -> None:
        if isinstance(l, list):
            ls = l
        elif isinstance(l, int):
            ls = [l]
        else:
            ls = list(l)

        self._ls_list = ls
        self._lmax = max(ls)
        self.normalization = normalization

    def __call__(self, alpha: mx.array, beta: mx.array) -> mx.array:
        y = mx.cos(beta)
        z = mx.sin(beta)
        sha = spherical_harmonics_alpha(self._lmax, alpha.flatten())  # [z, m]
        shy = legendre(self._ls_list, y.flatten(), z.flatten())  # [z, l * m]
        out = _mul_m_lm([(1, l) for l in self._ls_list], sha, shy)

        if self.normalization == "norm":
            norm_factors = []
            for l in self._ls_list:
                factor = math.sqrt(2 * l + 1) / math.sqrt(4 * math.pi)
                norm_factors.append(mx.ones(2 * l + 1, dtype=out.dtype) * factor)
            out = out / mx.concatenate(norm_factors)
        elif self.normalization == "component":
            out = out * math.sqrt(4 * math.pi)

        return out.reshape(alpha.shape + (shy.shape[1],))


def spherical_harmonics_alpha_beta(l, alpha, beta, *, normalization: str = "integral"):
    r"""Spherical harmonics of :math:`\vec r = R_y(\alpha) R_x(\beta) e_y`

    .. math:: Y^l(\alpha, \beta) = S^l(\alpha) P^l(\cos(\beta))

    where :math:`P^l` are the `Legendre` polynomials


    Parameters
    ----------
    l : int or list of int
        degree of the spherical harmonics.

    alpha : `mx.array`
        tensor of shape ``(...)``.

    beta : `mx.array`
        tensor of shape ``(...)``.

    Returns
    -------
    `mx.array`
        a tensor of shape ``(..., 2l+1)``
    """
    sh = SphericalHarmonicsAlphaBeta(l, normalization=normalization)
    return sh(alpha, beta)


def spherical_harmonics_alpha(l: int, alpha: mx.array) -> mx.array:
    r""":math:`S^l(\alpha)` of `spherical_harmonics_alpha_beta`

    Parameters
    ----------
    l : int
        degree of the spherical harmonics.

    alpha : `mx.array`
        tensor of shape ``(...)``.

    Returns
    -------
    `mx.array`
        a tensor of shape ``(..., 2l+1)``
    """
    alpha = alpha[..., None]  # [..., 1]
    m = mx.arange(1, l + 1, dtype=alpha.dtype)  # [1, 2, 3, ..., l]
    cos = mx.cos(m * alpha)  # [..., m]

    m = mx.arange(l, 0, -1, dtype=alpha.dtype)  # [l, l-1, l-2, ..., 1]
    sin = mx.sin(m * alpha)  # [..., m]

    out = mx.concatenate(
        [
            math.sqrt(2) * sin,
            mx.ones_like(alpha),
            math.sqrt(2) * cos,
        ],
        axis=-1,
    )

    return out  # [..., m]


def legendre(ls, y, z):
    """Compute Legendre polynomials for given ls."""
    out = mx.zeros(y.shape + (sum(2 * l + 1 for l in ls),), dtype=y.dtype)
    
    i = 0
    for l in ls:
        leg = []
        for m in range(l + 1):
            p = _poly_legendre(l, m)
            x = mx.zeros_like(y)
            
            for (zn, yn), c in p.items():
                x = x + float(c) * (y ** zn) * (z ** yn)
            
            leg.append(x[..., None])
        
        for m in range(-l, l + 1):
            out[..., i] = leg[abs(m)].squeeze(-1)
            i += 1
    
    return out


def _poly_legendre(l, m):
    r"""
    polynomial coefficients of legendre

    y = sqrt(1 - z^2)
    """
    z_sym, y_sym = symbols("z y", real=True)
    return Poly(_sympy_legendre(l, m), domain="R", gens=(z_sym, y_sym)).as_dict()


def _sympy_legendre(l, m) -> float:
    r"""
    en.wikipedia.org/wiki/Associated_Legendre_polynomials
    - remove two times (-1)^m
    - use another normalization such that P(l, -m) = P(l, m)
    - remove (-1)^l

    y = sqrt(1 - z^2)
    """
    l = Integer(l)
    m = Integer(abs(m))
    z_sym, y_sym = symbols("z y", real=True)
    ex = 1 / (2**l * factorial(l)) * y_sym**m * diff((z_sym**2 - 1) ** l, z_sym, l + m)
    ex *= sqrt((2 * l + 1) / (4 * pi) * factorial(l - m) / factorial(l + m))
    return ex


def _mul_m_lm(mul_l: List[Tuple[int, int]], x_m: mx.array, x_lm: mx.array) -> mx.array:
    """
    multiply tensor [..., l * m] by [..., m]
    """
    l_max = x_m.shape[-1] // 2
    out = []
    i = 0
    for mul, l in mul_l:
        d = mul * (2 * l + 1)
        x1 = x_lm[..., i : i + d]  # [..., mul * m]
        x1 = x1.reshape(x1.shape[:-1] + (mul, 2 * l + 1))  # [..., mul, m]
        x2 = x_m[..., l_max - l : l_max + l + 1]  # [..., m]
        x2 = x2.reshape(x2.shape[:-1] + (1, 2 * l + 1))  # [..., mul=1, m]
        x = x1 * x2
        x = x.reshape(x.shape[:-2] + (d,))
        out.append(x)
        i += d
    return mx.concatenate(out, axis=-1)