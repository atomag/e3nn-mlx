import mlx.core as mx
from mlx import nn

from .. import o3


class Identity(nn.Module):
    r"""Identity operation

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`

    irreps_out : `e3nn.o3.Irreps`
    """

    def __init__(self, irreps_in, irreps_out) -> None:
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in).simplify()
        self.irreps_out = o3.Irreps(irreps_out).simplify()

        assert self.irreps_in == self.irreps_out

        output_mask = mx.concatenate([mx.ones(mul * (2 * l + 1)) for mul, (l, _p) in self.irreps_out])
        self.output_mask = output_mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.irreps_in} -> {self.irreps_out})"

    def __call__(
        self,
        features,
    ):
        """evaluate"""
        return features