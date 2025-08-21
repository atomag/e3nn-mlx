from mlx import nn
import mlx.core as mx

from .gate_points_2101 import Convolution, Network as GatePointsNetwork

class SimpleNetwork(nn.Module):
    """Minimal stub simple network to satisfy imports.

    This placeholder returns zeros; replace with a proper implementation as needed.
    """
    def __init__(self, output_dim: int = 1) -> None:
        super().__init__()
        self.output_dim = int(output_dim)

    def __call__(self, data):
        n = data["pos"].shape[0] if "pos" in data else 1
        return mx.zeros((n, self.output_dim))

__all__ = [
    "Convolution",
    "GatePointsNetwork",
    "SimpleNetwork",
]
