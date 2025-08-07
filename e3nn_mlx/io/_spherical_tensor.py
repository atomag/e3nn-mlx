from math import pi
from e3nn_mlx.util._array_workarounds import array_at_set_workaround, spherical_harmonics_set_workaround

from collections import namedtuple
from typing import Tuple

import numpy as np
import mlx.core as mx
from scipy.signal import find_peaks

from e3nn_mlx import o3


def _find_peaks_2d(x):
    """Find peaks in 2D array using scipy"""
    x_np = np.array(x)
    iii = []
    for i in range(x_np.shape[0]):
        jj, _ = find_peaks(x_np[i, :])
        iii += [(i, j) for j in jj]

    jjj = []
    for j in range(x_np.shape[1]):
        ii, _ = find_peaks(x_np[:, j])
        jjj += [(i, j) for i in ii]

    return list(set(iii).intersection(set(jjj)))


class SphericalTensor(o3.Irreps):
    r"""representation of a signal on the sphere

    A `SphericalTensor` contains the coefficients :math:`A^l` of a function :math:`f` defined on the sphere

    .. math::
        f(x) = \sum_{l=0}^{l_\mathrm{max}} A^l \cdot Y^l(x)


    The way this function is transformed by parity :math:`f \longrightarrow P f` is described by the two parameters :math:`p_v`
    and :math:`p_a`

    .. math::
        (P f)(x) &= p_v f(p_a x)

        &= \sum_{l=0}^{l_\mathrm{max}} p_v p_a^l A^l \cdot Y^l(x)


    Parameters
    ----------
    lmax : int
        :math:`l_\mathrm{max}`

    p_val : {+1, -1}
        :math:`p_v`

    p_arg : {+1, -1}
        :math:`p_a`


    Examples
    --------

    >>> SphericalTensor(3, 1, 1)
    1x0e+1x1e+1x2e+1x3e

    >>> SphericalTensor(3, 1, -1)
    1x0e+1x1o+1x2e+1x3o
    """
    # pylint: disable=abstract-method

    def __new__(
        # pylint: disable=signature-differs
        cls,
        lmax,
        p_val,
        p_arg,
    ):
        return super().__new__(cls, [(1, (l, p_val * p_arg**l)) for l in range(lmax + 1)])

    def with_peaks_at(self, vectors, values=None):
        r"""Create a spherical tensor with peaks

        The peaks are located in :math:`\vec r_i` and have amplitude :math:`\|\vec r_i \|`

        Parameters
        ----------
        vectors : `mlx.array`
            :math:`\vec r_i` tensor of shape ``(N, 3)``

        values : `mlx.array`, optional
            value on the peak, tensor of shape ``(N)``

        Returns
        -------
        `mlx.array`
            tensor of shape ``(self.dim,)``

        Examples
        --------
        >>> s = SphericalTensor(4, 1, -1)
        >>> pos = mx.array([
        ...     [1.0, 0.0, 0.0],
        ...     [3.0, 4.0, 0.0],
        ... ])
        >>> x = s.with_peaks_at(pos)
        >>> s.signal_xyz(x, pos).astype(mx.int64)
        array([1, 5])

        >>> val = mx.array([
        ...     -1.5,
        ...     2.0,
        ... ])
        >>> x = s.with_peaks_at(pos, val)
        >>> s.signal_xyz(x, pos)
        array([-1.5000,  2.0000])
        """
        if values is not None:
            # Ensure broadcasting works correctly
            vectors = mx.broadcast_to(vectors, values.shape + (3,))
            values = mx.array(values)

        # empty set of vectors returns a 0 spherical tensor
        if vectors.size == 0:
            return mx.zeros(vectors.shape[:-2] + (self.dim,))

        assert (
            self[0][1].p == 1
        ), "since the value is set by the radii who is even, p_val has to be 1"  # pylint: disable=no-member

        assert vectors.ndim == 2 and vectors.shape[1] == 3

        if values is None:
            values = mx.linalg.norm(vectors, axis=1)  # [batch]
        
        # Filter out zero vectors
        mask = values != 0
        vectors = vectors[mask]
        values = values[mask]

        coeff = o3.spherical_harmonics(self, vectors, normalize=True)  # [batch, l * m]
        A = coeff.T @ coeff  # [dim, dim]
        # Y(v_a) . Y(v_b) solution_b = radii_a
        solution = mx.linalg.solve(A, coeff.T @ values)  # [dim]
        
        # Verify solution accuracy
        residual = mx.abs(values - coeff @ solution).max()
        assert residual < 1e-5 * mx.abs(values).max(), f"Solution accuracy: {residual}"

        return solution @ coeff.T

    def sum_of_diracs(self, positions: mx.array, values: mx.array) -> mx.array:
        r"""Sum (almost-) dirac deltas

        .. math::

            f(x) = \sum_i v_i \delta^L(\vec r_i)

        where :math:`\delta^L` is the apporximation of a dirac delta.

        Parameters
        ----------
        positions : `mlx.array`
            :math:`\vec r_i` tensor of shape ``(..., N, 3)``

        values : `mlx.array`
            :math:`v_i` tensor of shape ``(..., N)``

        Returns
        -------
        `mlx.array`
            tensor of shape ``(..., self.dim)``

        Examples
        --------
        >>> s = SphericalTensor(7, 1, -1)
        >>> pos = mx.array([
        ...     [1.0, 0.0, 0.0],
        ...     [0.0, 1.0, 0.0],
        ... ])
        >>> val = mx.array([
        ...     -1.0,
        ...     1.0,
        ... ])
        >>> x = s.sum_of_diracs(pos, val)
        >>> s.signal_xyz(x, mx.eye(3)).multiply(10.0).round()
        array([-10.,  10.,  -0.])

        >>> s.sum_of_diracs(mx.empty((1, 0, 2, 3)), mx.empty((2, 0, 1))).shape
        (2, 0, 64)

        >>> s.sum_of_diracs(mx.random.normal((1, 3, 2, 3)), mx.random.normal((2, 1, 1))).shape
        (2, 3, 64)
        """
        # Ensure broadcasting works correctly
        positions = mx.array(positions)
        values = mx.array(values)
        
        # Handle broadcasting
        positions_shape = positions.shape
        values_shape = values.shape
        
        if positions.numel() == 0:
            return mx.zeros(values_shape[:-1] + (self.dim,))

        y = o3.spherical_harmonics(self, positions, True)  # [..., N, dim]
        v = values[..., None]

        return 4 * pi / (self.lmax + 1) ** 2 * mx.sum(y * v, axis=-2)

    def from_samples_on_s2(self, positions: mx.array, values: mx.array, res: int = 100) -> mx.array:
        r"""Convert a set of position on the sphere and values into a spherical tensor

        Parameters
        ----------
        positions : `mlx.array`
            tensor of shape ``(..., N, 3)``

        values : `mlx.array`
            tensor of shape ``(..., N)``

        Returns
        -------
        `mlx.array`
            tensor of shape ``(..., self.dim)``

        Examples
        --------
        >>> s = SphericalTensor(2, 1, 1)
        >>> pos = mx.array([
        ...     [
        ...         [0.0, 0.0, 1.0],
        ...         [0.0, 0.0, -1.0],
        ...     ],
        ...     [
        ...         [0.0, 1.0, 0.0],
        ...         [0.0, -1.0, 0.0],
        ...     ],
        ... ], dtype=mx.float64)
        >>> val = mx.array([
        ...     [
        ...         1.0,
        ...         -1.0,
        ...     ],
        ...     [
        ...         1.0,
        ...         -1.0,
        ...     ],
        ... ], dtype=mx.float64)
        >>> s.from_samples_on_s2(pos, val, res=200).astype(mx.int64)
        array([[0, 0, 0, 3, 0, 0, 0, 0, 0],
               [0, 0, 3, 0, 0, 0, 0, 0, 0]])

        >>> pos = mx.empty((2, 0, 10, 3))
        >>> val = mx.empty((2, 0, 10))
        >>> s.from_samples_on_s2(pos, val)
        array([], shape=(2, 0, 9))
        """
        positions = mx.array(positions)
        values = mx.array(values)
        
        # Handle broadcasting
        positions_shape = positions.shape
        values_shape = values.shape
        
        if positions.numel() == 0:
            return mx.zeros(values_shape[:-1] + (self.dim,))

        positions = mx.linalg.normalize(positions, axis=-1)  # forward 0's instead of nan for zero-radius

        size = positions.shape[:-2]
        n = positions.shape[-2]
        positions = positions.reshape(-1, n, 3)
        values = values.reshape(-1, n)

        s2 = o3.FromS2Grid(res=res, lmax=self.lmax, normalization="integral", dtype=values.dtype)
        pos = s2.grid.reshape(1, -1, 3)

        # Compute pairwise distances
        cd = mx.linalg.norm(pos[:, :, None] - positions[:, None, :], axis=-1)  # [batch, b*a, N]
        i = mx.arange(len(values)).reshape(-1, 1)  # [batch, 1]
        j = mx.argmin(cd, axis=2)  # [batch, b*a]
        val = values[i, j]  # [batch, b*a]
        val = val.reshape(*size, s2.res_beta, s2.res_alpha)

        return s2(val)

    def norms(self, signal) -> mx.array:
        r"""The norms of each l component

        Parameters
        ----------
        signal : `mlx.array`
            tensor of shape ``(..., dim)``

        Returns
        -------
        `mlx.array`
            tensor of shape ``(..., lmax+1)``

        Examples
        --------
        >>> s = SphericalTensor(1, 1, -1)
        >>> s.norms(mx.array([1.5, 0.0, 3.0, 4.0]))
        array([1.5000, 5.0000])
        """
        i = 0
        norms = []
        for _, ir in self:
            norms += [mx.linalg.norm(signal[..., i : i + ir.dim], axis=-1)]
            i += ir.dim
        return mx.stack(norms, axis=-1)

    def signal_xyz(self, signal, r) -> mx.array:
        r"""Evaluate the signal on given points on the sphere

        .. math::

            f(\vec x / \|\vec x\|)

        Parameters
        ----------
        signal : `mlx.array`
            tensor of shape ``(*A, self.dim)``

        r : `mlx.array`
            tensor of shape ``(*B, 3)``

        Returns
        -------
        `mlx.array`
            tensor of shape ``(*A, *B)``

        Examples
        --------
        >>> s = SphericalTensor(3, 1, -1)
        >>> s.signal_xyz(s.randn(2, 1, 3, -1), mx.random.normal((2, 4, 3))).shape
        (2, 1, 3, 2, 4)
        """
        sh = o3.spherical_harmonics(self, r, normalize=True)
        dim = (self.lmax + 1) ** 2
        output = mx.einsum("bi,ai->ab", sh.reshape(-1, dim), signal.reshape(-1, dim))
        return output.reshape(signal.shape[:-1] + r.shape[:-1])

    def signal_on_grid(self, signal, res: int = 100, normalization: str = "integral"):
        r"""Evaluate the signal on a grid on the sphere"""
        Ret = namedtuple("Return", "grid, values")
        s2 = o3.ToS2Grid(lmax=self.lmax, res=res, normalization=normalization)
        return Ret(s2.grid, s2(signal))

    def plotly_surface(
        self, signals, centers=None, res: int = 100, radius: bool = True, relu: bool = False, normalization: str = "integral"
    ):
        r"""Create traces for plotly

        Examples
        --------
        >>> import plotly.graph_objects as go
        >>> x = SphericalTensor(4, +1, +1)
        >>> traces = x.plotly_surface(x.randn(-1))
        >>> traces = [go.Surface(**d) for d in traces]
        >>> fig = go.Figure(data=traces)
        """
        signals = signals.reshape(-1, self.dim)

        if centers is None:
            centers = [None] * len(signals)
        else:
            centers = centers.reshape(-1, 3)

        traces = []
        for signal, center in zip(signals, centers):
            r, f = self.plot(signal, center, res, radius, relu, normalization)
            traces += [
                {
                    "x": np.array(r[:, :, 0]),
                    "y": np.array(r[:, :, 1]),
                    "z": np.array(r[:, :, 2]),
                    "surfacecolor": np.array(f),
                }
            ]
        return traces

    def plot(
        self, signal, center=None, res: int = 100, radius: bool = True, relu: bool = False, normalization: str = "integral"
    ) -> Tuple[mx.array, mx.array]:
        r"""Create surface in order to make a plot"""
        assert signal.ndim == 1

        r, f = self.signal_on_grid(signal, res, normalization)
        f = mx.maximum(f, 0) if relu else f

        # beta: [0, pi]
        r = mx.array(r)
        r = array_at_set_workaround(r, 0, mx.array([0.0, 1.0, 0.0]))
        r = array_at_set_workaround(r, -1, mx.array([0.0, -1.0, 0.0]))
        f = mx.array(f)
        f = array_at_set_workaround(f, 0, mx.mean(f[0]))
        f = array_at_set_workaround(f, -1, mx.mean(f[-1]))

        # alpha: [0, 2pi]
        r = mx.concatenate([r, r[:, :1]], axis=1)  # [beta, alpha, 3]
        f = mx.concatenate([f, f[:, :1]], axis=1)  # [beta, alpha]

        if radius:
            r *= mx.abs(f)[..., None]

        if center is not None:
            r += center

        return r, f

    def find_peaks(self, signal, res: int = 100) -> Tuple[mx.array, mx.array]:
        r"""Locate peaks on the sphere

        Examples
        --------
        >>> s = SphericalTensor(4, 1, -1)
        >>> pos = mx.array([
        ...     [4.0, 0.0, 4.0],
        ...     [0.0, 5.0, 0.0],
        ... ])
        >>> x = s.with_peaks_at(pos)
        >>> pos, val = s.find_peaks(x)
        >>> pos[val > 4.0].multiply(10).round().abs()
        array([[ 7.,  0.,  7.],
               [ 0., 10.,  0.]])
        >>> val[val > 4.0].multiply(10).round().abs()
        array([57., 50.])
        """
        x1, f1 = self.signal_on_grid(signal, res)

        abc = mx.array([pi / 2, pi / 2, pi / 2])
        R = o3.angles_to_matrix(*abc)
        D = self.D_from_matrix(R)

        r_signal = D @ signal
        rx2, f2 = self.signal_on_grid(r_signal, res)
        x2 = mx.einsum("ij,baj->bai", R.T, rx2)

        # Convert to numpy for peak finding
        f1_np = np.array(f1)
        ij = _find_peaks_2d(f1_np)
        if len(ij) > 0:
            x1p = mx.stack([x1[i, j] for i, j in ij])
            f1p = mx.stack([f1[i, j] for i, j in ij])
        else:
            x1p = mx.empty((0, 3))
            f1p = mx.empty(0)

        f2_np = np.array(f2)
        ij = _find_peaks_2d(f2_np)
        if len(ij) > 0:
            x2p = mx.stack([x2[i, j] for i, j in ij])
            f2p = mx.stack([f2[i, j] for i, j in ij])
        else:
            x2p = mx.empty((0, 3))
            f2p = mx.empty(0)

        # Union of the results
        if len(x1p) > 0 and len(x2p) > 0:
            mask = mx.linalg.norm(x1p[:, None] - x2p[None, :], axis=-1) < 2 * pi / res
            mask_sum = mx.sum(mask, axis=1)
            x = mx.concatenate([x1p[mask_sum == 0], x2p])
            f = mx.concatenate([f1p[mask_sum == 0], f2p])
        elif len(x1p) > 0:
            x = x1p
            f = f1p
        elif len(x2p) > 0:
            x = x2p
            f = f2p
        else:
            x = mx.empty((0, 3))
            f = mx.empty(0)

        return x, f