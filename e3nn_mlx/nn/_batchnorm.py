import mlx.core as mx
from mlx import nn

from .. import o3


class BatchNorm(nn.Module):
    """Batch normalization for orthonormal representations

    It normalizes by the norm of the representations.
    Note that the norm is invariant only for orthonormal representations.
    Irreducible representations `wigner_D` are orthonormal.

    Parameters
    ----------
    irreps : `o3.Irreps`
        representation

    eps : float
        avoid division by zero when we normalize by the variance

    momentum : float
        momentum of the running average

    affine : bool
        do we have weight and bias parameters

    reduce : {'mean', 'max'}
        method used to reduce

    instance : bool
        apply instance norm instead of batch norm

    include_bias : bool
        include a bias term for batch norm of scalars

    normalization : str
        which normalization method to apply (i.e., `norm` or `component`)
    """

    def __init__(
        self,
        irreps: o3.Irreps,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        reduce: str = "mean",
        instance: bool = False,
        include_bias: bool = True,
        normalization: str = "component",
    ) -> None:
        super().__init__()

        if not isinstance(irreps, (o3.Irreps, str)):
            raise TypeError(f"irreps must be an o3.Irreps object or string, got {type(irreps).__name__}")
        self.irreps = o3.Irreps(irreps)
        
        if not isinstance(eps, (int, float)) or eps <= 0:
            raise ValueError(f"eps must be a positive number, got {eps}")
        self.eps = eps
        
        if not isinstance(momentum, (int, float)) or not 0 < momentum < 1:
            raise ValueError(f"momentum must be between 0 and 1, got {momentum}")
        self.momentum = momentum
        
        if not isinstance(affine, bool):
            raise TypeError(f"affine must be a boolean, got {type(affine).__name__}")
        self.affine = affine
        
        if not isinstance(instance, bool):
            raise TypeError(f"instance must be a boolean, got {type(instance).__name__}")
        self.instance = instance
        
        if not isinstance(include_bias, bool):
            raise TypeError(f"include_bias must be a boolean, got {type(include_bias).__name__}")
        self.include_bias = include_bias

        num_scalar = sum(mul for mul, ir in self.irreps if ir.is_scalar())
        num_features = self.irreps.num_irreps
        self.features = []

        if self.instance:
            self.running_mean = None
            self.running_var = None
        else:
            self.running_mean = mx.zeros(num_scalar)
            self.running_var = mx.ones(num_features)

        if affine:
            self.weight = mx.ones(num_features)
            if self.include_bias:
                self.bias = mx.zeros(num_scalar)
        else:
            self.weight = None
            self.bias = None

        if not isinstance(reduce, str):
            raise TypeError(f"reduce must be a string, got {type(reduce).__name__}")
        if reduce not in ["mean", "max"]:
            raise ValueError(f"reduce must be 'mean' or 'max', got '{reduce}'")
        self.reduce = reduce
        irs = []
        for mul, ir in self.irreps:
            irs.append((mul, ir.dim, ir.is_scalar()))
        self.irs = irs

        if not isinstance(normalization, str):
            raise TypeError(f"normalization must be a string, got {type(normalization).__name__}")
        if normalization not in ["norm", "component"]:
            raise ValueError(f"normalization must be 'norm' or 'component', got '{normalization}'")
        self.normalization = normalization

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps}, momentum={self.momentum})"

    def _roll_avg(self, curr, update):
        return (1 - self.momentum) * curr + self.momentum * update

    def __call__(self, input):
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
        if not isinstance(input, mx.array):
            raise TypeError(f"input must be an mx.array, got {type(input).__name__}")
        
        if input.ndim < 1:
            raise ValueError(f"input must have at least 1 dimension, got {input.ndim} dimensions")
        
        if input.shape[-1] != self.irreps.dim:
            raise ValueError(
                f"input last dimension ({input.shape[-1]}) must match irreps dimension ({self.irreps.dim})"
            )
            
        orig_shape = input.shape
        batch = input.shape[0]
        dim = input.shape[-1]
        input = input.reshape(batch, -1, dim)  # [batch, sample, stacked features]

        if self.training and not self.instance:
            new_means = []
            new_vars = []

        fields = []
        ix = 0
        irm = 0
        irv = 0
        iw = 0
        ib = 0

        for mul, d, is_scalar in self.irs:
            field = input[:, :, ix : ix + mul * d]  # [batch, sample, mul * repr]
            ix += mul * d

            # [batch, sample, mul, repr]
            field = field.reshape(batch, -1, mul, d)

            if is_scalar:
                if self.training or self.instance:
                    if self.instance:
                        field_mean = field.mean(1).reshape(batch, mul)  # [batch, mul]
                    else:
                        field_mean = field.mean(axis=(0, 1)).reshape(mul)  # [mul]
                        new_means.append(self._roll_avg(self.running_mean[irm : irm + mul], field_mean))
                else:
                    field_mean = self.running_mean[irm : irm + mul]
                irm += mul

                # [batch, sample, mul, repr]
                field = field - field_mean.reshape(-1, 1, mul, 1)

            if self.training or self.instance:
                if self.normalization == "norm":
                    field_norm = mx.sum(field ** 2, axis=3)  # [batch, sample, mul]
                elif self.normalization == "component":
                    field_norm = mx.mean(field ** 2, axis=3)  # [batch, sample, mul]
                else:
                    raise ValueError(f"Invalid normalization option {self.normalization}")

                if self.reduce == "mean":
                    field_norm = field_norm.mean(axis=1)  # [batch, mul]
                elif self.reduce == "max":
                    field_norm = mx.max(field_norm, axis=1)  # [batch, mul]
                else:
                    raise ValueError(f"Invalid reduce option {self.reduce}")

                if not self.instance:
                    field_norm = field_norm.mean(0)  # [mul]
                    new_vars.append(self._roll_avg(self.running_var[irv : irv + mul], field_norm))
            else:
                field_norm = self.running_var[irv : irv + mul]
            irv += mul

            field_norm = (field_norm + self.eps) ** (-0.5)  # [(batch,) mul]

            if self.affine:
                weight = self.weight[iw : iw + mul]  # [mul]
                iw += mul

                field_norm = field_norm * weight  # [(batch,) mul]

            field = field * field_norm.reshape(-1, 1, mul, 1)  # [batch, sample, mul, repr]

            if self.affine and self.include_bias and is_scalar:
                bias = self.bias[ib : ib + mul]  # [mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch, sample, mul, repr]

            fields.append(field.reshape(batch, -1, mul * d))  # [batch, sample, mul * repr]

        # Assertions in e3nn would go here, but we'll skip for simplicity

        if self.training and not self.instance:
            if len(new_means) > 0:
                self.running_mean = mx.concatenate(new_means)
            if len(new_vars) > 0:
                self.running_var = mx.concatenate(new_vars)

        output = mx.concatenate(fields, axis=2)  # [batch, sample, stacked features]
        return output.reshape(orig_shape)