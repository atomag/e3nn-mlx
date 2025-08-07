import mlx.core as mx
import mlx.nn as nn

from ._irreps import Irreps
from ._tensor_product._tensor_product import TensorProduct


class Norm(nn.Module):
    r"""Norm of each irrep in a direct sum of irreps.

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        representation of the input

    squared : bool, optional
        Whether to return the squared norm. ``False`` by default, i.e. the norm itself (sqrt of squared norm) is returned.

    Examples
    --------
    Compute the norms of 17 vectors.

    >>> norm = Norm("17x1o")
    >>> norm(mx.random.normal((17 * 3,))).shape
    (17,)
    """

    squared: bool

    def __init__(self, irreps_in, squared: bool = False) -> None:
        super().__init__()

        irreps_in = Irreps(irreps_in).simplify()
        irreps_out = Irreps([(mul, "0e") for mul, _ in irreps_in])

        instr = [(i, i, i, "uuu", False, ir.dim) for i, (mul, ir) in enumerate(irreps_in)]

        self.tp = TensorProduct(irreps_in, irreps_in, irreps_out, instr, irrep_normalization="component")

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out.simplify()
        self.squared = squared

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.irreps_in})"

    def __call__(self, features):
        """Compute norms of irreps in ``features``.

        Parameters
        ----------
        features : `mx.array`
            tensor of shape ``(..., irreps_in.dim)``

        Returns
        -------
        `mx.array`
            tensor of shape ``(..., irreps_out.dim)``
        """
        out = self.tp(features, features)
        if self.squared:
            return out
        else:
            # Use ReLU equivalent and sqrt for numerical stability
            return mx.sqrt(mx.maximum(out, 0.0))