from typing import Optional

import mlx.core as mx

from e3nn_mlx.o3._irreps import Irreps
from e3nn_mlx.o3._reduce import ReducedTensorProducts


class CartesianTensor(Irreps):
    r"""representation of a cartesian tensor into irreps

    Parameters
    ----------
    formula : str

    Examples
    --------

    >>> import mlx.core as mx
    >>> CartesianTensor("ij=-ji")
    1x1e

    >>> x = CartesianTensor("ijk=-jik=-ikj")
    >>> x.from_cartesian(mx.ones((3, 3, 3)))
    array([0.])

    >>> x.from_vectors(mx.ones(3), mx.ones(3), mx.ones(3))
    array([0.])

    >>> x = CartesianTensor("ij=ji")
    >>> t = mx.arange(9, dtype=mx.float32).reshape(3,3)
    >>> y = x.from_cartesian(t)
    >>> z = x.to_cartesian(y)
    >>> mx.allclose(z, (t + t.T)/2, atol=1e-5)
    array(True)
    """

    # pylint: disable=abstract-method

    # These are set in __new__
    formula: str
    indices: str

    def __new__(
        # pylint: disable=signature-differs
        cls,
        formula,
    ):
        indices = formula.split("=")[0].replace("-", "")
        rtp = ReducedTensorProducts(formula, **{i: "1o" for i in indices})
        ret = super().__new__(cls, rtp.irreps_out)
        ret.formula = formula
        ret.indices = indices
        return ret

    def from_cartesian(self, data, rtp=None):
        r"""convert cartesian tensor into irreps

        Parameters
        ----------
        data : `mlx.array`
            cartesian tensor of shape ``(..., 3, 3, 3, ...)``

        Returns
        -------
        `mlx.array`
            irreps tensor of shape ``(..., self.dim)``
        """
        if rtp is None:
            rtp = self.reduced_tensor_products(data)

        Q = rtp.change_of_basis.flatten(-len(self.indices))
        return data.flatten(-len(self.indices)) @ Q.T

    def from_vectors(self, *xs, rtp=None):
        r"""convert :math:`x_1 \otimes x_2 \otimes x_3 \otimes \dots`

        Parameters
        ----------
        xs : list of `mlx.array`
            list of vectors of shape ``(..., 3)``

        Returns
        -------
        `mlx.array`
            irreps tensor of shape ``(..., self.dim)``
        """
        if rtp is None:
            rtp = self.reduced_tensor_products(xs[0])

        return rtp(*xs)  # pylint: disable=not-callable

    def to_cartesian(self, data, rtp=None):
        r"""convert irreps tensor to cartesian tensor

        This is the symmetry-aware inverse operation of ``from_cartesian()``.

        Parameters
        ----------
        data : `mlx.array`
            irreps tensor of shape ``(..., D)``, where D is the dimension of the irreps,
            i.e. ``D=self.dim``.

        Returns
        -------
        `mlx.array`
            cartesian tensor of shape ``(..., 3, 3, 3, ...)``
        """
        if rtp is None:
            rtp = self.reduced_tensor_products(data)

        Q = rtp.change_of_basis
        cartesian_tensor = data @ Q.flatten(-len(self.indices))

        shape = list(data.shape[:-1]) + list(Q.shape[1:])
        cartesian_tensor = cartesian_tensor.reshape(shape)

        return cartesian_tensor

    def reduced_tensor_products(self, data: Optional[mx.array] = None) -> ReducedTensorProducts:
        r"""reduced tensor products

        Returns
        -------
        `e3nn_mlx.ReducedTensorProducts`
            reduced tensor products
        """
        rtp = ReducedTensorProducts(self.formula, **{i: "1o" for i in self.indices})
        if data is not None:
            rtp = rtp.to(device=None, dtype=data.dtype)
        return rtp