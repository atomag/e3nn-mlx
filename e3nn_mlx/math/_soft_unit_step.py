import mlx.core as mx


def soft_unit_step(x):
    r"""smooth :math:`C^\infty` version of the unit step function

    .. math::

        x \mapsto \theta(x) e^{-1/x}


    Parameters
    ----------
    x : `mx.array`
        tensor of shape :math:`(...)`

    Returns
    -------
    `mx.array`
        tensor of shape :math:`(...)`

    Examples
    --------

    .. jupyter-execute::
        :hide-code:

        import mlx.core as mx
        from e3nn_mlx.math import soft_unit_step
        import matplotlib.pyplot as plt

    .. jupyter-execute::

        x = mx.linspace(-1.0, 10.0, 1000)
        plt.plot(x, soft_unit_step(x));
    """
    # MLX doesn't have custom autograd functions like PyTorch
    # We'll implement the forward pass directly (placeholder)
    y = mx.zeros_like(x)
    mask = x > 0.0
    y = mx.where(mask, mx.exp(-1 / x), y)
    return y
