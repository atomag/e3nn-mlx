import mlx.core as mx
import mlx.nn as nn

from ._wigner import wigner_D
from ._s2grid import _quadrature_weights, s2_grid
# angles_to_xyz function not available in simplified rotation module
# from ._rotation import angles_to_xyz


def flat_wigner(lmax: int, alpha: mx.array, beta: mx.array, gamma: mx.array) -> mx.array:
    """Flatten Wigner D-matrices for all l up to lmax."""
    return mx.concatenate([(2 * l + 1) ** 0.5 * wigner_D(l, alpha, beta, gamma).flatten(-2) for l in range(lmax + 1)], axis=-1)


class SO3Grid(nn.Module):
    r"""Apply non linearity on the signal on SO(3)

    Parameters
    ----------
    lmax : int
        irreps representation ``[(2 * l + 1, (l, p_val)) for l in [0, ..., lmax]]``

    resolution : int
        SO(3) grid resolution

    normalization : {'norm', 'component'}

    aspect_ratio : float
        default value (2) should be optimal
    """

    def __init__(self, lmax, resolution, *, normalization: str = "component", aspect_ratio: int = 2) -> None:
        super().__init__()

        assert normalization == "component"

        nb = 2 * resolution
        na = round(2 * aspect_ratio * resolution)

        b, a = s2_grid(nb, na)
        self.D = flat_wigner(lmax, a[:, None, None], b[None, :, None], a[None, None, :])
        qw = _quadrature_weights(nb // 2) * nb**2 / na**2
        self.qw = qw

        self.alpha = a
        self.beta = b
        self.gamma = a

        self.res_alpha = na
        self.res_beta = nb
        self.res_gamma = na

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.lmax})"

    def to_grid(self, features) -> mx.array:
        r"""evaluate

        Parameters
        ----------

        features : `mx.array`
            tensor of shape ``(..., self.irreps.dim)``

        Returns
        -------
        `mx.array`
            tensor of shape ``(..., self.res_alpha, self.res_beta, self.res_gamma)``
        """
        return mx.einsum("...i,abci->...abc", features, self.D) / self.D.shape[-1] ** 0.5

    def from_grid(self, features) -> mx.array:
        r"""evaluate

        Parameters
        ----------

        features : `mx.array`
            tensor of shape ``(..., self.res_alpha, self.res_beta, self.res_gamma)``

        Returns
        -------
        `mx.array`
            tensor of shape ``(..., self.irreps.dim)``
        """
        return mx.einsum("...abc,abci,b->...i", features, self.D, self.qw) * self.D.shape[-1] ** 0.5