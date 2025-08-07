import math
import mlx.core as mx

from ._soft_unit_step import soft_unit_step


def soft_one_hot_linspace(x: mx.array, start, end, number, basis=None, cutoff=None) -> mx.array:
    r"""Projection on a basis of functions

    Returns a set of :math:`\{y_i(x)\}_{i=1}^N`,

    .. math::

        y_i(x) = \frac{1}{Z} f_i(x)

    where :math:`x` is the input and :math:`f_i` is the ith basis function.
    :math:`Z` is a constant defined (if possible) such that,

    .. math::

        \langle \sum_{i=1}^N y_i(x)^2 \rangle_x \approx 1

    See the last plot below.
    Note that ``bessel`` basis cannot be normalized.

    Parameters
    ----------
    x : `mx.array`
        tensor of shape :math:`(...)`

    start : float
        minimum value span by the basis

    end : float
        maximum value span by the basis

    number : int
        number of basis functions :math:`N`

    basis : {'gaussian', 'cosine', 'smooth_finite', 'fourier', 'bessel'}
        choice of basis family; note that due to the :math:`1/x` term, ``bessel`` basis does not satisfy the normalization of
        other basis choices

    cutoff : bool
        if ``cutoff=True`` then for all :math:`x` outside of the interval defined by ``(start, end)``,
        :math:`\forall i, \; f_i(x) \approx 0`

    Returns
    -------
    `mx.array`
        tensor of shape :math:`(..., N)`

    Examples
    --------

    .. jupyter-execute::
        :hide-code:

        import mlx.core as mx
        from e3nn_mlx.math import soft_one_hot_linspace
        import matplotlib.pyplot as plt

    .. jupyter-execute::

        bases = ['gaussian', 'cosine', 'smooth_finite', 'fourier', 'bessel']
        x = mx.linspace(-1.0, 2.0, 100)

    .. jupyter-execute::

        fig, axss = plt.subplots(len(bases), 2, figsize=(9, 6), sharex=True, sharey=True)

        for axs, b in zip(axss, bases):
            for ax, c in zip(axs, [True, False]):
                plt.sca(ax)
                plt.plot(x, soft_one_hot_linspace(x, -0.5, 1.5, number=4, basis=b, cutoff=c))
                plt.plot([-0.5]*2, [-2, 2], 'k-.')
                plt.plot([1.5]*2, [-2, 2], 'k-.')
                plt.title(f"{b}" + (" with cutoff" if c else ""))

        plt.ylim(-1, 1.5)
        plt.tight_layout()

    .. jupyter-execute::

        fig, axss = plt.subplots(len(bases), 2, figsize=(9, 6), sharex=True, sharey=True)

        for axs, b in zip(axss, bases):
            for ax, c in zip(axs, [True, False]):
                plt.sca(ax)
                plt.plot(x, soft_one_hot_linspace(x, -0.5, 1.5, number=4, basis=b, cutoff=c).pow(2).sum(1))
                plt.plot([-0.5]*2, [-2, 2], 'k-.')
                plt.plot([1.5]*2, [-2, 2], 'k-.')
                plt.title(f"{b}" + (" with cutoff" if c else ""))

        plt.ylim(0, 2)
        plt.tight_layout()
    """
    # pylint: disable=misplaced-comparison-comparison

    # Input validation
    if not isinstance(x, mx.array):
        raise TypeError(f"x must be an mx.array, got {type(x).__name__}")
    
    if not isinstance(start, (int, float)):
        raise TypeError(f"start must be a number, got {type(start).__name__}")
    
    if not isinstance(end, (int, float)):
        raise TypeError(f"end must be a number, got {type(end).__name__}")
    
    if not isinstance(number, int) or number <= 0:
        raise ValueError(f"number must be a positive integer, got {number}")
    
    if basis is not None and not isinstance(basis, str):
        raise TypeError(f"basis must be a string or None, got {type(basis).__name__}")
    
    if cutoff not in [True, False]:
        raise ValueError("cutoff must be specified as True or False")
    
    if start >= end:
        raise ValueError(f"start ({start}) must be less than end ({end})")
    
    valid_bases = ['gaussian', 'cosine', 'smooth_finite', 'fourier', 'bessel']
    if basis is not None and basis not in valid_bases:
        raise ValueError(f"basis must be one of {valid_bases}, got '{basis}'")

    if not cutoff:
        values = mx.linspace(start, end, number, dtype=x.dtype)
        step = values[1] - values[0]
    else:
        values = mx.linspace(start, end, number + 2, dtype=x.dtype)
        step = values[1] - values[0]
        values = values[1:-1]

    diff = (x[..., None] - values) / step

    if basis == "gaussian":
        return mx.exp(-diff**2) / 1.12

    if basis == "cosine":
        return mx.cos(math.pi / 2 * diff) * (diff < 1) * (-1 < diff)

    if basis == "smooth_finite":
        return 1.14136 * mx.exp(mx.array(2.0)) * soft_unit_step(diff + 1) * soft_unit_step(1 - diff)

    if basis == "fourier":
        x_norm = (x[..., None] - start) / (end - start)
        if not cutoff:
            i = mx.arange(0, number, dtype=x.dtype)
            return mx.cos(math.pi * i * x_norm) / math.sqrt(0.25 + number / 2)
        else:
            i = mx.arange(1, number + 1, dtype=x.dtype)
            return mx.sin(math.pi * i * x_norm) / math.sqrt(0.25 + number / 2) * (0 < x_norm) * (x_norm < 1)

    if basis == "bessel":
        x_bessel = x[..., None] - start
        c = end - start
        bessel_roots = mx.arange(1, number + 1, dtype=x.dtype) * math.pi
        
        # Handle potential division by zero
        try:
            out = math.sqrt(2 / c) * mx.sin(bessel_roots * x_bessel / c) / x_bessel
        except Exception as e:
            # Replace infinities with zeros (limit as x_bessel -> 0)
            out = math.sqrt(2 / c) * mx.sin(bessel_roots * x_bessel / c) / mx.maximum(x_bessel, 1e-8)
            out = mx.where(x_bessel == 0, math.sqrt(2 / c) * bessel_roots / c, out)

        if not cutoff:
            return out
        else:
            return out * ((x_bessel / c) < 1) * (0 < x_bessel)

    raise ValueError(f'basis="{basis}" is not a valid entry. Valid options are: {valid_bases}')