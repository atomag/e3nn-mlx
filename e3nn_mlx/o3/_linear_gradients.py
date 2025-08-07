import mlx.core as mx
from typing import Optional, Dict, Any
from e3nn_mlx.o3._linear_equivariant import LinearEquivariant


class LinearWithGradients(LinearEquivariant):
    """
    Linear layer with gradient computation support for training.
    
    This extends the LinearEquivariant class to provide gradient
    computation capabilities similar to PyTorch's autograd.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradients: Dict[str, Optional[mx.array]] = {'weight': None, 'bias': None}
        self._input_cache: Optional[mx.array] = None
        self._weight_cache: Optional[mx.array] = None
        self._bias_cache: Optional[mx.array] = None

    def __call__(self, features: mx.array, weight: Optional[mx.array] = None, bias: Optional[mx.array] = None) -> mx.array:
        """
        Forward pass with input caching for gradient computation.
        """
        # Cache inputs for gradient computation
        self._input_cache = features
        self._weight_cache = weight if weight is not None else self.weight
        self._bias_cache = bias if bias is not None else self.bias
        
        # Call parent forward pass
        return super().__call__(features, weight, bias)

    def backward(self, grad_output: mx.array) -> None:
        """
        Compute gradients for weight and bias parameters.
        
        Parameters
        ----------
        grad_output : mx.array
            Gradient of the loss with respect to the output
        """
        if self._input_cache is None:
            raise RuntimeError("Forward pass must be called before backward pass")
        
        batch_size = self._input_cache.shape[0]
        
        # Compute weight gradients
        if self.weight_numel > 0 and self._weight_cache.size > 0:
            self.gradients['weight'] = self._compute_weight_gradient(grad_output, batch_size)
        
        # Compute bias gradients
        if self.bias_numel > 0 and self._bias_cache.size > 0:
            self.gradients['bias'] = self._compute_bias_gradient(grad_output, batch_size)

    def _compute_weight_gradient(self, grad_output: mx.array, batch_size: int) -> mx.array:
        """
        Compute gradients for weights.
        
        This implements the gradient computation for the linear transformation:
        dL/dW = sum_{batch} (dL/dY * X^T)
        """
        # Reshape grad_output for proper computation
        if self.f_out is None:
            grad_reshaped = grad_output.reshape(batch_size, -1)
        else:
            grad_reshaped = grad_output.reshape(batch_size, self.f_out, -1)
        
        # Reshape input for proper computation
        if self.f_in is None:
            input_reshaped = self._input_cache.reshape(batch_size, -1)
        else:
            input_reshaped = self._input_cache.reshape(batch_size, self.f_in, -1)
        
        # Initialize weight gradient
        weight_grad = mx.zeros_like(self._weight_cache)
        
        # Compute gradients for each instruction
        weight_offset = 0
        for pattern in self.weight_patterns:
            ins = pattern['instruction']
            path_size = prod(ins.path_shape)
            
            # Extract relevant parts of input and output gradient
            input_slice = input_reshaped[..., pattern['input_slice']]
            
            if self.f_out is None:
                grad_slice = grad_reshaped[..., pattern['output_slice']]
            else:
                grad_slice = grad_reshaped[..., :, pattern['output_slice']]
            
            # Compute gradient for this weight block
            if self.shared_weights:
                if self.f_in is None and self.f_out is None:
                    # Simple case: sum over batch
                    block_grad = mx.einsum('zi,zj->ij', input_slice, grad_slice)
                elif self.f_in is not None and self.f_out is not None:
                    # Both f_in and f_out present
                    block_grad = mx.einsum('zfi,zfj->fij', input_slice, grad_slice)
                elif self.f_in is not None:
                    # Only f_in present
                    block_grad = mx.einsum('zfi,zj->fij', input_slice, grad_slice)
                else:
                    # Only f_out present
                    block_grad = mx.einsum('zi,zfj->fij', input_slice, grad_slice)
            else:
                # Non-shared weights (per-batch gradients)
                if self.f_in is None and self.f_out is None:
                    block_grad = mx.einsum('zi,zj->zij', input_slice, grad_slice)
                elif self.f_in is not None and self.f_out is not None:
                    block_grad = mx.einsum('zfi,zfj->zfij', input_slice, grad_slice)
                elif self.f_in is not None:
                    block_grad = mx.einsum('zfi,zj->zfij', input_slice, grad_slice)
                else:
                    block_grad = mx.einsum('zi,zfj->zfij', input_slice, grad_slice)
            
            # Apply path weight to gradient
            block_grad = block_grad * ins.path_weight
            
            # Store in weight gradient
            weight_slice = weight_grad[weight_offset:weight_offset + path_size]
            weight_slice[:] = block_grad.reshape(-1)
            
            weight_offset += path_size
        
        return weight_grad

    def _compute_bias_gradient(self, grad_output: mx.array, batch_size: int) -> mx.array:
        """
        Compute gradients for bias terms.
        
        This implements the gradient computation for biases:
        dL/db = sum_{batch} (dL/dY)
        """
        # Initialize bias gradient
        bias_grad = mx.zeros_like(self._bias_cache)
        
        # Sum gradients over batch for each bias instruction
        bias_offset = 0
        for ins in self.instructions:
            if ins.i_in == -1:  # Bias instruction
                mul_ir_out = self.irreps_out[ins.i_out]
                
                # Extract relevant part of gradient
                if self.f_out is None:
                    grad_slice = grad_output[..., 
                        sum(ir.dim for ir in self.irreps_out[:ins.i_out]):
                        sum(ir.dim for ir in self.irreps_out[:ins.i_out + 1])
                    ]
                else:
                    grad_slice = grad_output[..., :, 
                        sum(ir.dim for ir in self.irreps_out[:ins.i_out]):
                        sum(ir.dim for ir in self.irreps_out[:ins.i_out + 1])
                    ]
                
                # Sum over batch dimensions
                if self.f_out is None:
                    bias_slice_grad = mx.sum(grad_slice, axis=tuple(range(grad_slice.ndim - 1)))
                else:
                    bias_slice_grad = mx.sum(grad_slice, axis=tuple(range(grad_slice.ndim - 2)))
                
                # Apply path weight
                bias_slice_grad = bias_slice_grad * ins.path_weight
                
                # Store in bias gradient
                path_size = prod(ins.path_shape)
                bias_slice = bias_grad[bias_offset:bias_offset + path_size]
                bias_slice[:] = bias_slice_grad
                
                bias_offset += path_size
        
        return bias_grad

    def get_gradients(self) -> Dict[str, Optional[mx.array]]:
        """
        Get computed gradients.
        
        Returns
        -------
        Dict[str, Optional[mx.array]]
            Dictionary containing 'weight' and 'bias' gradients
        """
        return self.gradients.copy()

    def zero_grad(self) -> None:
        """
        Zero out all gradients.
        """
        self.gradients = {'weight': None, 'bias': None}
        self._input_cache = None
        self._weight_cache = None
        self._bias_cache = None

    def step(self, learning_rate: float = 0.01) -> None:
        """
        Perform a gradient descent step on the parameters.
        
        Parameters
        ----------
        learning_rate : float
            Learning rate for the update
        """
        if self.gradients['weight'] is not None and self.internal_weights:
            self.weight = self.weight - learning_rate * self.gradients['weight']
        
        if self.gradients['bias'] is not None and self.internal_weights:
            self.bias = self.bias - learning_rate * self.gradients['bias']

    def get_parameter_norms(self) -> Dict[str, float]:
        """
        Get the L2 norms of parameters and their gradients.
        
        Returns
        -------
        Dict[str, float]
            Dictionary with 'weight_norm', 'bias_norm', 'weight_grad_norm', 'bias_grad_norm'
        """
        norms = {}
        
        if self.weight.size > 0:
            norms['weight_norm'] = float(mx.linalg.norm(self.weight))
            if self.gradients['weight'] is not None:
                norms['weight_grad_norm'] = float(mx.linalg.norm(self.gradients['weight']))
        
        if self.bias.size > 0:
            norms['bias_norm'] = float(mx.linalg.norm(self.bias))
            if self.gradients['bias'] is not None:
                norms['bias_grad_norm'] = float(mx.linalg.norm(self.gradients['bias']))
        
        return norms


class Optimizer:
    """
    Simple optimizer for linear layers with gradient computation.
    """
    
    def __init__(self, parameters: list, learning_rate: float = 0.01):
        self.parameters = parameters
        self.learning_rate = learning_rate
    
    def step(self) -> None:
        """
        Perform optimization step for all parameters.
        """
        for param in self.parameters:
            if hasattr(param, 'gradients') and param.gradients.get('weight') is not None:
                param.weight = param.weight - self.learning_rate * param.gradients['weight']
            if hasattr(param, 'gradients') and param.gradients.get('bias') is not None:
                param.bias = param.bias - self.learning_rate * param.gradients['bias']
    
    def zero_grad(self) -> None:
        """
        Zero gradients for all parameters.
        """
        for param in self.parameters:
            if hasattr(param, 'zero_grad'):
                param.zero_grad()


def create_training_example():
    """
    Create a simple training example to demonstrate gradient computation.
    """
    from e3nn_mlx.o3._irreps import Irreps
    
    # Create a simple linear layer
    irreps_in = Irreps("2x0e + 1x1o")
    irreps_out = Irreps("1x0e + 1x1o")
    
    linear = LinearWithGradients(
        irreps_in, 
        irreps_out, 
        biases=True,
        weight_init="xavier_normal"
    )
    
    # Create some training data
    batch_size = 4
    x = mx.random.normal((batch_size, irreps_in.dim))
    y_target = mx.random.normal((batch_size, irreps_out.dim))
    
    # Forward pass
    y_pred = linear(x)
    
    # Compute loss (MSE)
    loss = mx.mean((y_pred - y_target) ** 2)
    
    print(f"Initial loss: {float(loss):.6f}")
    
    # Backward pass (manual gradient computation for demonstration)
    loss_grad = 2.0 * (y_pred - y_target) / batch_size
    linear.backward(loss_grad)
    
    # Check gradients
    gradients = linear.get_gradients()
    norms = linear.get_parameter_norms()
    
    print(f"Weight norm: {norms.get('weight_norm', 0):.6f}")
    print(f"Bias norm: {norms.get('bias_norm', 0):.6f}")
    print(f"Weight grad norm: {norms.get('weight_grad_norm', 0):.6f}")
    print(f"Bias grad norm: {norms.get('bias_grad_norm', 0):.6f}")
    
    # Gradient descent step
    linear.step(learning_rate=0.1)
    
    # Forward pass again
    y_pred_new = linear(x)
    loss_new = mx.mean((y_pred_new - y_target) ** 2)
    
    print(f"Loss after update: {float(loss_new):.6f}")
    
    return linear, loss, loss_new


if __name__ == "__main__":
    # Run the training example
    create_training_example()