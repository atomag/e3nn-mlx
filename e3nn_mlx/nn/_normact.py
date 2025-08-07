from typing import Callable, Optional

import mlx.core as mx
from mlx import nn

from ..o3._irreps import Irreps
from ..o3._norm import Norm
from ..o3._tensor_product._sub import ElementwiseTensorProduct


class NormActivation(nn.Module):
    r"""Norm-based activation function
    Applies a scalar nonlinearity to the norm of each irrep and ouputs a (normalized) version of that irrep multiplied by the
    scalar output of the scalar nonlinearity.
    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        representation of the input
    scalar_nonlinearity : callable
        scalar nonlinearity such as ``mx.sigmoid``
    normalize : bool
        whether to normalize the input features before multiplying them by the scalars from the nonlinearity
    epsilon : float, optional
        when ``normalize``ing, norms smaller than ``epsilon`` will be clamped up to ``epsilon`` to avoid division by zero and
        NaN gradients. Not allowed when ``normalize`` is False.
    bias : bool
        whether to apply a learnable additive bias to the inputs of the ``scalar_nonlinearity``
    Examples
    --------
    >>> n = NormActivation("2x1e", mx.sigmoid)
    >>> feats = mx.ones((1, 2*3))
    >>> print(feats.reshape(1, 2, 3).norm(axis=-1))
    mx.array([[1.7320509 1.7320509]])
    >>> print(mx.sigmoid(feats.reshape(1, 2, 3).norm(axis=-1)))
    mx.array([[0.84974027 0.84974027]])
    >>> print(n(feats).reshape(1, 2, 3).norm(axis=-1))
    mx.array([[0.84974027 0.84974027]])
    """

    epsilon: Optional[float]
    _eps_squared: float

    def __init__(
        self,
        irreps_in: Irreps,
        scalar_nonlinearity: Callable,
        normalize: bool = True,
        epsilon: Optional[float] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_in)

        if epsilon is None and normalize:
            epsilon = 1e-8
        elif epsilon is not None and not normalize:
            raise ValueError("epsilon and normalize = False don't make sense together")
        elif not epsilon > 0:
            raise ValueError(f"epsilon {epsilon} is invalid, must be strictly positive.")
        self.epsilon = epsilon
        if self.epsilon is not None:
            self._eps_squared = epsilon * epsilon
        else:
            self._eps_squared = 0.0  # doesn't matter

        # if we have an epsilon, use squared and do the sqrt ourselves
        self.norm = Norm(irreps_in, squared=(epsilon is not None))
        self.scalar_nonlinearity = scalar_nonlinearity
        self.normalize = normalize
        self.bias = bias
        if self.bias:
            self.biases = mx.zeros(irreps_in.num_irreps)

        self.scalar_multiplier = ElementwiseTensorProduct(
            irreps_in1=self.norm.irreps_out,
            irreps_in2=irreps_in,
        )

    def __call__(self, features):
        """evaluate
        Parameters
        ----------
        features : `mlx.core.array`
            tensor of shape ``(..., irreps_in.dim)``
        Returns
        -------
        `mlx.core.array`
            tensor of shape ``(..., irreps_in.dim)``
        """
        norms = self.norm(features)
        if self._eps_squared > 0:
            # See TFN for the original version of this approach:
            # https://github.com/tensorfieldnetworks/tensorfieldnetworks/blob/master/tensorfieldnetworks/utils.py#L22
            norms = mx.where(norms < self._eps_squared, self._eps_squared, norms)
            norms = mx.sqrt(norms)

        nonlin_arg = norms
        if self.bias:
            nonlin_arg = nonlin_arg + self.biases

        scalings = self.scalar_nonlinearity(nonlin_arg)
        if self.normalize:
            scalings = scalings / norms

        return self.scalar_multiplier(scalings, features)