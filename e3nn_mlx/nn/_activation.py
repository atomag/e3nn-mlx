import mlx.core as mx
from mlx import nn

from ..o3._irreps import Irreps
from ..math._normalize_activation import normalize2mom


class Activation(nn.Module):
    r"""Scalar activation function.

    Odd scalar inputs require activation functions with a defined parity (odd or even).

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        representation of the input

    acts : list of function or None
        list of activation functions, `None` if non-scalar or identity

    Examples
    --------

    >>> a = Activation("256x0o", [mx.abs])
    >>> a.irreps_out
    256x0e

    >>> a = Activation("256x0o+16x1e", [None, None])
    >>> a.irreps_out
    256x0o+16x1e
    """

    def __init__(self, irreps_in, acts) -> None:
        super().__init__()
        irreps_in = Irreps(irreps_in)
        if len(irreps_in) != len(acts):
            raise ValueError(f"Irreps in and number of activation functions does not match: {len(acts), (irreps_in, acts)}")

        # normalize the second moment - wrap activation functions
        def make_normalized_activation(act):
            if act is None:
                return None
            def normalized_act(x):
                return normalize2mom(act(x))
            return normalized_act
        
        acts = [make_normalized_activation(act) for act in acts]

        irreps_out = []
        for (mul, (l_in, p_in)), act in zip(irreps_in, acts):
            if act is not None:
                if l_in != 0:
                    raise ValueError("Activation: cannot apply an activation function to a non-scalar input.")

                x = mx.linspace(0, 10, 256)

                a1, a2 = act(x), act(-x)
                if mx.max(mx.abs(a1 - a2)) < 1e-5:
                    p_act = 1
                elif mx.max(mx.abs(a1 + a2)) < 1e-5:
                    p_act = -1
                else:
                    p_act = 0

                p_out = p_act if p_in == -1 else p_in
                irreps_out.append((mul, (0, p_out)))

                if p_out == 0:
                    raise ValueError(
                        "Activation: the parity is violated! The input scalar is odd but the activation is neither "
                        "even nor odd."
                    )
            else:
                irreps_out.append((mul, (l_in, p_in)))

        self.irreps_in = irreps_in
        self.irreps_out = Irreps(irreps_out)
        self.acts = acts
        self.paths = [(mul, (l, p), act) for (mul, (l, p)), act in zip(self.irreps_in, self.acts)]
        assert len(self.irreps_in) == len(self.acts)

    def __repr__(self) -> str:
        acts = "".join(["x" if a is not None else " " for a in self.acts])
        return f"{self.__class__.__name__} [{acts}] ({self.irreps_in} -> {self.irreps_out})"

    def __call__(self, features, dim: int = -1):
        """evaluate

        Parameters
        ----------
        features : `mlx.core.array`
            tensor of shape ``(...)``

        Returns
        -------
        `mlx.core.array`
            tensor of shape the same shape as the input
        """
        output = []
        index = 0
        for mul, (l, _), act in self.paths:
            ir_dim = 2 * l + 1
            if act is not None:
                # Extract scalar features and apply activation
                scalar_features = mx.take(features, mx.arange(index, index + mul), axis=dim)
                activated = act(scalar_features)
                output.append(activated)
            else:
                # Keep non-scalar features unchanged
                non_scalar_features = mx.take(features, mx.arange(index, index + mul * ir_dim), axis=dim)
                output.append(non_scalar_features)
            index += mul * ir_dim

        if len(output) > 1:
            return mx.concatenate(output, axis=dim)
        elif len(output) == 1:
            return output[0]
        else:
            return mx.zeros_like(features)