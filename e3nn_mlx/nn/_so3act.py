import mlx.core as mx
from mlx import nn

from ..math._normalize_activation import normalize2mom
from ..o3 import SO3Grid


class SO3Activation(nn.Module):
    r"""Apply non linearity on the signal on SO(3)

    Parameters
    ----------
    lmax_in : int
        input lmax

    lmax_out : int
        output lmax

    act : function
        activation function :math:`\phi`

    resolution : int
        SO(3) grid resolution

    normalization : {'norm', 'component'}
    """

    def __init__(self, lmax_in, lmax_out, act, resolution, *, normalization: str = "component", aspect_ratio: int = 2) -> None:
        super().__init__()

        self.grid_in = SO3Grid(lmax_in, resolution, normalization=normalization, aspect_ratio=aspect_ratio)
        self.grid_out = SO3Grid(lmax_out, resolution, normalization=normalization, aspect_ratio=aspect_ratio)
        self.act = normalize2mom(act)

        self.lmax_in = lmax_in
        self.lmax_out = lmax_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.lmax_in} -> {self.lmax_out})"

    def __call__(self, features):
        r"""evaluate

        Parameters
        ----------

        features : `mlx.core.array`
            tensor of shape ``(..., self.irreps_in.dim)``

        Returns
        -------
        `mlx.core.array`
            tensor of shape ``(..., self.irreps_out.dim)``
        """
        features = self.grid_in.to_grid(features)
        features = self.act(features)
        features = self.grid_out.from_grid(features)

        return features