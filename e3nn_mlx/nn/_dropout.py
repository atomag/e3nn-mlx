import mlx.core as mx
from mlx import nn

from .. import o3


class Dropout(nn.Module):
    """Equivariant Dropout

    :math:`A_{zai}` is the input and :math:`B_{zai}` is the output where

    - ``z`` is the batch index
    - ``a`` any non-batch and non-irrep index
    - ``i`` is the irrep index, for instance if ``irreps="0e + 2x1e"`` then ``i=2`` select the *second vector*

    .. math::

        B_{zai} = \frac{x_{zi}}{1-p} A_{zai}

    where :math:`p` is the dropout probability and :math:`x` is a Bernoulli random variable with parameter :math:`1-p`.

    Parameters
    ----------
    irreps : `o3.Irreps`
        representation

    p : float
        probability to drop
    """

    def __init__(self, irreps, p) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.p = p

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.irreps}, p={self.p})"

    def __call__(self, x):
        """evaluate

        Parameters
        ----------
        input : `mlx.core.array`
            tensor of shape ``(batch, ..., irreps.dim)``

        Returns
        -------
        `mlx.core.array`
            tensor of shape ``(batch, ..., irreps.dim)``
        """
        if not self.training:
            return x

        batch = x.shape[0]

        noises = []
        for mul, (l, _p) in self.irreps:
            dim = 2 * l + 1
            
            if self.p >= 1:
                noise = mx.zeros((batch, mul))
            elif self.p <= 0:
                noise = mx.ones((batch, mul))
            else:
                # Generate Bernoulli random variable and scale
                noise = mx.random.bernoulli(1 - self.p, (batch, mul)) / (1 - self.p)

            noise = mx.repeat(noise[:, :, None], dim, axis=2).reshape(batch, mul * dim)
            noises.append(noise)

        noise = mx.concatenate(noises, axis=-1)
        while noise.ndim < x.ndim:
            noise = noise[:, None]
        return x * noise
