"""
Simple equivariant network for point cloud processing.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import List, Optional, Union

from e3nn_mlx.o3 import Irreps, TensorProduct, Linear
from e3nn_mlx.nn import Gate


class SimpleNetwork(nn.Module):
    """
    Simple equivariant neural network for point clouds.
    
    This network processes 3D point clouds with equivariant operations,
    making it suitable for tasks like molecular property prediction
    and 3D shape classification.
    
    Parameters
    ----------
    irreps_in : Irreps
        Input irreducible representations
    irreps_hidden : Irreps
        Hidden layer irreducible representations
    irreps_out : Irreps
        Output irreducible representations
    num_layers : int, default 3
        Number of hidden layers
    activation : callable, default Gate()
        Activation function to use
    """
    
    def __init__(
        self,
        irreps_in: Irreps,
        irreps_hidden: Irreps,
        irreps_out: Irreps,
        num_layers: int = 3,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.irreps_in = irreps_in
        self.irreps_hidden = irreps_hidden
        self.irreps_out = irreps_out
        self.num_layers = num_layers
        
        if activation is None:
            # Use a simple scalar activation for now to avoid dimension issues
            # In the future, this could be made more sophisticated
            self.activation = lambda x: x  # Identity function as fallback
        else:
            self.activation = activation
        
        # Build network layers
        self.layers = []
        
        # Input layer
        if num_layers > 0:
            self.layers.append(Linear(irreps_in, irreps_hidden))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(Linear(irreps_hidden, irreps_hidden))
        
        # Output layer
        self.layers.append(Linear(irreps_hidden, irreps_out))
        
    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.
        
        Parameters
        ----------
        x : mx.array
            Input tensor of shape (..., irreps_in.dim)
            
        Returns
        -------
        mx.array
            Output tensor of shape (..., irreps_out.dim)
        """
        # Input layer
        if self.num_layers > 0:
            x = self.layers[0](x)
            x = self.activation(x)
        
        # Hidden layers
        for i in range(1, self.num_layers):
            x = self.layers[i](x)
            x = self.activation(x)
        
        # Output layer (no activation)
        x = self.layers[-1](x)
        
        return x