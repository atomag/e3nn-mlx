import mlx.core as mx
import math
from typing import Optional, Union, List, Tuple, Any
from e3nn_mlx.o3._irreps import Irreps
from e3nn_mlx.util import prod


class LinearEquivariant:
    """
    Complete equivariant linear layer implementation for MLX
    matching PyTorch e3nn behavior.
    
    This implementation addresses the key differences identified between
    PyTorch and MLX linear layers:
    - Proper weight initialization schemes
    - Equivariance constraints
    - Correct handling of irreducible representations
    - Path normalization matching PyTorch behavior
    """

    def __init__(
        self,
        irreps_in: Union[str, Irreps],
        irreps_out: Union[str, Irreps],
        *,
        f_in: Optional[int] = None,
        f_out: Optional[int] = None,
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        instructions: Optional[List[Tuple[int, int]]] = None,
        biases: Union[bool, List[bool]] = False,
        path_normalization: str = "element",
        weight_init: str = "xavier_normal",
        bias_init: str = "zeros"
    ):
        # Convert to Irreps if needed
        self.irreps_in = Irreps(irreps_in) if isinstance(irreps_in, str) else irreps_in
        self.irreps_out = Irreps(irreps_out) if isinstance(irreps_out, str) else irreps_out
        
        assert path_normalization in ["element", "path"]
        self.path_normalization = path_normalization
        
        # Process instructions
        if instructions is None:
            # By default, make all possible connections
            instructions = [
                (i_in, i_out)
                for i_in, (_, ir_in) in enumerate(self.irreps_in)
                for i_out, (_, ir_out) in enumerate(self.irreps_out)
                if ir_in == ir_out
            ]
        
        # Create instruction objects with path normalization
        self.instructions = self._create_instructions(instructions)
        
        # Process biases
        if biases is None:
            biases = len(self.irreps_out) * (False,)
        if isinstance(biases, bool):
            biases = [biases and ir.is_scalar() for _, ir in self.irreps_out]
        
        assert len(biases) == len(self.irreps_out)
        assert all(ir.is_scalar() or (not b) for b, (_, ir) in zip(biases, self.irreps_out))
        
        # Add bias instructions
        self.instructions += [
            type('Instruction', (), {
                'i_in': -1,
                'i_out': i_out,
                'path_shape': (mul_ir.dim,),
                'path_weight': 1.0
            })()
            for i_out, (bias, mul_ir) in enumerate(zip(biases, self.irreps_out))
            if bias
        ]
        
        # Process weight arguments
        if shared_weights is False and internal_weights is None:
            internal_weights = False
        
        if shared_weights is None:
            shared_weights = True
        
        if internal_weights is None:
            internal_weights = True
        
        assert shared_weights or not internal_weights
        self.internal_weights = internal_weights
        self.shared_weights = shared_weights
        self.f_in = f_in
        self.f_out = f_out
        
        # Calculate weight and bias dimensions
        self.weight_numel, self.bias_numel = self._compute_weight_bias_numel()
        
        # Initialize weights with proper initialization
        self.weight = self._initialize_weights(weight_init)
        
        # Initialize biases
        self.bias = self._initialize_biases(bias_init)
        
        # Compute output mask
        self.output_mask = self._compute_output_mask()
        
        # Precompute weight patterns for efficient access
        self.weight_patterns = self._compute_weight_patterns()

    def _create_instructions(self, instructions: List[Tuple[int, int]]) -> List[Any]:
        """Create instruction objects with proper path normalization."""
        base_instructions = [
            type('Instruction', (), {
                'i_in': i_in,
                'i_out': i_out,
                'path_shape': (self.irreps_in[i_in].mul, self.irreps_out[i_out].mul),
                'path_weight': 1.0,
            })()
            for i_in, i_out in instructions
        ]
        
        # Apply path normalization (matching PyTorch behavior)
        def alpha(ins) -> float:
            x = sum(
                self.irreps_in[i.i_in if self.path_normalization == "element" else ins.i_in].mul
                for i in base_instructions
                if i.i_out == ins.i_out
            )
            if self.f_in is not None:
                x *= self.f_in
            return 1.0 if x == 0 else x
        
        normalized_instructions = []
        for ins in base_instructions:
            normalized_ins = type('Instruction', (), {
                'i_in': ins.i_in,
                'i_out': ins.i_out,
                'path_shape': ins.path_shape,
                'path_weight': alpha(ins) ** (-0.5)
            })()
            normalized_instructions.append(normalized_ins)
        
        return normalized_instructions

    def _compute_weight_bias_numel(self) -> Tuple[int, int]:
        """Compute the number of weight and bias parameters."""
        weight_numel = sum(prod(ins.path_shape) for ins in self.instructions if ins.i_in != -1)
        bias_numel = sum(self.irreps_out[ins.i_out].dim for ins in self.instructions if ins.i_in == -1)
        return weight_numel, bias_numel

    def _initialize_weights(self, init_method: str) -> mx.array:
        """Initialize weights using specified method matching PyTorch patterns."""
        if self.weight_numel == 0:
            return mx.array([])
        
        if not self.internal_weights:
            return mx.array([])
        
        # Shape includes f_in and f_out dimensions if specified
        shape = ((self.f_in, self.f_out) if self.f_in is not None else ()) + (self.weight_numel,)
        
        if init_method == "xavier_normal":
            # Xavier/Glorot normal initialization
            fan_in = self.irreps_in.dim
            fan_out = self.irreps_out.dim
            if self.f_in is not None:
                fan_in *= self.f_in
            if self.f_out is not None:
                fan_out *= self.f_out
            
            scale = math.sqrt(2.0 / (fan_in + fan_out))
            weights = mx.random.normal(shape) * scale
        elif init_method == "xavier_uniform":
            # Xavier/Glorot uniform initialization
            fan_in = self.irreps_in.dim
            fan_out = self.irreps_out.dim
            if self.f_in is not None:
                fan_in *= self.f_in
            if self.f_out is not None:
                fan_out *= self.f_out
            
            scale = math.sqrt(6.0 / (fan_in + fan_out))
            weights = mx.random.uniform(-scale, scale, shape)
        elif init_method == "kaiming_normal":
            # Kaiming/He normal initialization
            fan_in = self.irreps_in.dim
            if self.f_in is not None:
                fan_in *= self.f_in
            
            scale = math.sqrt(2.0 / fan_in)
            weights = mx.random.normal(shape) * scale
        elif init_method == "normal":
            # Standard normal (matching PyTorch's default)
            weights = mx.random.normal(shape)
        else:
            raise ValueError(f"Unknown weight initialization method: {init_method}")
        
        return weights

    def _initialize_biases(self, init_method: str) -> mx.array:
        """Initialize bias terms."""
        if self.bias_numel == 0:
            return mx.array([])
        
        if not self.internal_weights:
            return mx.array([])
        
        shape = ((self.f_out,) if self.f_out is not None else ()) + (self.bias_numel,)
        
        if init_method == "zeros":
            return mx.zeros(shape)
        elif init_method == "ones":
            return mx.ones(shape)
        elif init_method == "normal":
            return mx.random.normal(shape)
        else:
            raise ValueError(f"Unknown bias initialization method: {init_method}")

    def _compute_output_mask(self) -> mx.array:
        """Compute output mask indicating which outputs are active."""
        if self.irreps_out.dim == 0:
            return mx.ones(0)
        
        output_mask = mx.concatenate([
            (
                mx.ones(mul_ir.dim)
                if any((ins.i_out == i_out) and (0 not in ins.path_shape) for ins in self.instructions)
                else mx.zeros(mul_ir.dim)
            )
            for i_out, mul_ir in enumerate(self.irreps_out)
        ])
        return output_mask

    def _compute_weight_patterns(self) -> List[dict]:
        """Compute weight reshaping patterns for each instruction."""
        patterns = []
        
        weight_offset = 0
        for ins in self.instructions:
            if ins.i_in != -1:  # Skip bias instructions
                pattern = {
                    'instruction': ins,
                    'weight_slice': slice(weight_offset, weight_offset + prod(ins.path_shape)),
                    'input_slice': slice(
                        sum(ir.dim for ir in self.irreps_in[:ins.i_in]),
                        sum(ir.dim for ir in self.irreps_in[:ins.i_in + 1])
                    ),
                    'output_slice': slice(
                        sum(ir.dim for ir in self.irreps_out[:ins.i_out]),
                        sum(ir.dim for ir in self.irreps_out[:ins.i_out + 1])
                    )
                }
                patterns.append(pattern)
                weight_offset += prod(ins.path_shape)
        
        return patterns

    def __call__(self, features: mx.array, weight: Optional[mx.array] = None, bias: Optional[mx.array] = None) -> mx.array:
        """
        Forward pass with proper equivariance handling.
        
        Parameters
        ----------
        features : mx.array
            Input tensor of shape (..., irreps_in.dim)
        weight : mx.array, optional
            External weights (if internal_weights=False)
        bias : mx.array, optional
            External biases (if internal_weights=False)
        
        Returns
        -------
        mx.array
            Output tensor of shape (..., irreps_out.dim)
        """
        if weight is None:
            if self.weight_numel > 0 and not self.internal_weights:
                raise RuntimeError("Weights must be provided when internal_weights = False")
            weight = self.weight
        if bias is None:
            if self.bias_numel > 0 and not self.internal_weights:
                raise RuntimeError("Biases must be provided when internal_weights = False")
            bias = self.bias
        
        return self._forward(features, weight, bias)

    def _forward(self, x: mx.array, weights: mx.array, biases: mx.array) -> mx.array:
        """Main forward pass implementation."""
        # Handle input shape
        original_shape = x.shape
        batch_shape = original_shape[:-1]
        
        # Handle empty input case
        if x.shape[-1] == 0:
            return mx.zeros(batch_shape + (self.irreps_out.dim,), dtype=x.dtype)
        
        # Reshape input for processing
        if self.f_in is None:
            x_reshaped = x.reshape(-1, x.shape[-1])
        else:
            x_reshaped = x.reshape(-1, self.f_in, x.shape[-1])
        
        batch_size = x_reshaped.shape[0]
        
        # Extract input irreps
        input_list = self._extract_input_irreps(x_reshaped)
        
        # Process weights and biases
        weight_list = self._process_weights(weights)
        bias_list = self._process_biases(biases)
        
        # Compute outputs for each output irrep
        outputs = []
        for i_out, mul_ir_out in enumerate(self.irreps_out):
            output_parts = []
            
            # Add contributions from input instructions
            for pattern in weight_list:
                if pattern['instruction'].i_out == i_out:
                    input_idx = pattern['instruction'].i_in
                    if input_idx >= 0 and input_idx < len(input_list):
                        transformed = self._apply_linear_transformation(
                            input_list[input_idx], pattern, batch_size
                        )
                        output_parts.append(transformed)
            
            # Add bias contributions
            for bias_info in bias_list:
                if bias_info['instruction'].i_out == i_out:
                    bias_expanded = self._expand_bias(bias_info['bias'], batch_size)
                    output_parts.append(bias_expanded * bias_info['instruction'].path_weight)
            
            # Combine output parts
            if output_parts:
                output = sum(output_parts)
            else:
                output = mx.zeros((batch_size, mul_ir_out.dim))
            
            outputs.append(output)
        
        # Concatenate all outputs
        if outputs:
            result = mx.concatenate(outputs, axis=-1)
        else:
            result = mx.zeros((batch_size, self.irreps_out.dim))
        
        # Reshape to original batch shape
        if self.f_out is None:
            final_shape = batch_shape + (self.irreps_out.dim,)
        else:
            final_shape = batch_shape + (self.f_out, self.irreps_out.dim)
        
        return result.reshape(final_shape)

    def _extract_input_irreps(self, x_reshaped: mx.array) -> List[mx.array]:
        """Extract individual irreps from input tensor."""
        if len(self.irreps_in) == 1:
            return [x_reshaped.reshape(
                x_reshaped.shape[0], 
                *(() if self.f_in is None else (self.f_in,)), 
                self.irreps_in[0].mul, 
                self.irreps_in[0].ir.dim
            )]
        else:
            input_list = []
            start = 0
            for i, mul_ir in enumerate(self.irreps_in):
                end = start + mul_ir.dim
                input_slice = x_reshaped[..., start:end]
                if self.f_in is None:
                    reshaped_slice = input_slice.reshape(-1, mul_ir.mul, mul_ir.ir.dim)
                else:
                    reshaped_slice = input_slice.reshape(-1, self.f_in, mul_ir.mul, mul_ir.ir.dim)
                input_list.append(reshaped_slice)
                start = end
            return input_list

    def _process_weights(self, weights: mx.array) -> List[dict]:
        """Process weights into per-instruction format."""
        if self.weight_numel == 0 or weights.size == 0:
            return []
        
        weight_list = []
        for pattern in self.weight_patterns:
            ins = pattern['instruction']
            weight_slice = weights[pattern['weight_slice']]
            
            # Reshape weight for proper broadcasting
            if self.shared_weights:
                if self.f_in is None and self.f_out is None:
                    weight_reshaped = weight_slice.reshape(ins.path_shape)
                elif self.f_in is not None and self.f_out is not None:
                    weight_reshaped = weight_slice.reshape(self.f_in, self.f_out, *ins.path_shape)
                elif self.f_in is not None:
                    weight_reshaped = weight_slice.reshape(self.f_in, *ins.path_shape)
                else:
                    weight_reshaped = weight_slice.reshape(self.f_out, *ins.path_shape)
            else:
                # Non-shared weights have batch dimension
                if self.f_in is None and self.f_out is None:
                    weight_reshaped = weight_slice.reshape(-1, *ins.path_shape)
                elif self.f_in is not None and self.f_out is not None:
                    weight_reshaped = weight_slice.reshape(-1, self.f_in, self.f_out, *ins.path_shape)
                elif self.f_in is not None:
                    weight_reshaped = weight_slice.reshape(-1, self.f_in, *ins.path_shape)
                else:
                    weight_reshaped = weight_slice.reshape(-1, self.f_out, *ins.path_shape)
            
            weight_list.append({
                'instruction': ins,
                'weight': weight_reshaped,
                'input_slice': pattern['input_slice'],
                'output_slice': pattern['output_slice']
            })
        
        return weight_list

    def _process_biases(self, biases: mx.array) -> List[dict]:
        """Process biases into per-instruction format."""
        if self.bias_numel == 0 or biases.size == 0:
            return []
        
        bias_list = []
        bias_offset = 0
        
        for ins in self.instructions:
            if ins.i_in == -1:  # Bias instruction
                bias_slice = biases[..., bias_offset:bias_offset + prod(ins.path_shape)]
                bias_list.append({
                    'instruction': ins,
                    'bias': bias_slice
                })
                bias_offset += prod(ins.path_shape)
        
        return bias_list

    def _apply_linear_transformation(self, input_tensor: mx.array, weight_info: dict, batch_size: int) -> mx.array:
        """Apply linear transformation with proper einsum."""
        ins = weight_info['instruction']
        weight = weight_info['weight']
        
        # Determine einsum pattern based on dimensions
        if self.shared_weights:
            if self.f_in is None and self.f_out is None:
                # Simple case: no f_in/f_out dimensions
                einsum_pattern = "uw,zui->zwi"
            elif self.f_in is not None and self.f_out is not None:
                # Both f_in and f_out present
                einsum_pattern = "fufw,zfui->zfwi"
            elif self.f_in is not None:
                # Only f_in present
                einsum_pattern = "fuw,zfui->zwi"
            else:
                # Only f_out present
                einsum_pattern = "ufw,zui->zufwi"
        else:
            # Non-shared weights (batch dimension)
            if self.f_in is None and self.f_out is None:
                einsum_pattern = "zuw,zui->zwi"
            elif self.f_in is not None and self.f_out is not None:
                einsum_pattern = "zufufw,zfui->zfwi"
            elif self.f_in is not None:
                einsum_pattern = "zufuw,zfui->zwi"
            else:
                einsum_pattern = "zufw,zui->zufwi"
        
        # Apply einsum
        try:
            result = mx.einsum(einsum_pattern, weight, input_tensor)
        except Exception:
            # Fallback to simpler implementation if einsum fails
            result = self._fallback_linear_transformation(input_tensor, weight, ins)
        
        # Apply path weight
        result = result * ins.path_weight
        
        # Reshape to expected output format
        mul_ir_out = self.irreps_out[ins.i_out]
        if self.f_out is None:
            result = result.reshape(batch_size, mul_ir_out.dim)
        else:
            result = result.reshape(batch_size, self.f_out, mul_ir_out.dim)
        
        return result

    def _fallback_linear_transformation(self, input_tensor: mx.array, weight: mx.array, ins: Any) -> mx.array:
        """Fallback linear transformation for complex cases."""
        # Reshape for matrix multiplication
        batch_size = input_tensor.shape[0]
        
        # Flatten appropriate dimensions
        if self.f_in is None:
            input_flat = input_tensor.reshape(batch_size, -1)
        else:
            input_flat = input_tensor.reshape(batch_size * self.f_in, -1)
        
        if self.shared_weights:
            if self.f_in is None and self.f_out is None:
                weight_flat = weight.reshape(-1, input_flat.shape[-1])
            elif self.f_in is not None and self.f_out is not None:
                weight_flat = weight.reshape(self.f_in * self.f_out, -1)
            elif self.f_in is not None:
                weight_flat = weight.reshape(self.f_in, -1)
            else:
                weight_flat = weight.reshape(self.f_out, -1)
        else:
            # Non-shared weights
            weight_flat = weight.reshape(batch_size, -1)
        
        # Matrix multiplication
        result_flat = mx.matmul(input_flat, weight_flat.T)
        
        # Reshape back
        mul_ir_out = self.irreps_out[ins.i_out]
        if self.f_out is None:
            result = result_flat.reshape(batch_size, mul_ir_out.dim)
        else:
            result = result_flat.reshape(batch_size, self.f_out, mul_ir_out.dim)
        
        return result

    def _expand_bias(self, bias: mx.array, batch_size: int) -> mx.array:
        """Expand bias to match batch dimensions."""
        if self.f_out is None:
            return mx.broadcast_to(bias, (batch_size, bias.shape[-1]))
        else:
            return mx.broadcast_to(bias, (batch_size, self.f_out, bias.shape[-1]))

    def weight_view_for_instruction(self, instruction: int, weight: Optional[mx.array] = None) -> mx.array:
        """Get weight view for specific instruction."""
        if weight is None:
            if not self.internal_weights:
                raise RuntimeError("Weights must be provided when internal_weights = False")
            weight = self.weight
        
        if instruction >= len(self.weight_patterns):
            raise IndexError(f"Instruction {instruction} out of range")
        
        pattern = self.weight_patterns[instruction]
        weight_slice = weight[pattern['weight_slice']]
        return weight_slice.reshape(pattern['instruction'].path_shape)

    def weight_views(self, weight: Optional[mx.array] = None, yield_instruction: bool = False):
        """Iterator over weight views for all instructions."""
        if weight is None:
            if not self.internal_weights:
                raise RuntimeError("Weights must be provided when internal_weights = False")
            weight = self.weight
        
        for i, pattern in enumerate(self.weight_patterns):
            weight_view = self.weight_view_for_instruction(i, weight)
            if yield_instruction:
                yield i, pattern['instruction'], weight_view
            else:
                yield weight_view

    def __repr__(self) -> str:
        return f"LinearEquivariant({self.irreps_in} -> {self.irreps_out} | {self.weight_numel} weights)"


class PyTorchLinearWrapper:
    """
    Wrapper to make MLX linear layer behave like PyTorch linear layer
    for easier testing and compatibility.
    """
    
    def __init__(self, mlx_linear: LinearEquivariant):
        self.mlx_linear = mlx_linear
    
    def __call__(self, x: mx.array, weight: Optional[mx.array] = None, bias: Optional[mx.array] = None) -> mx.array:
        return self.mlx_linear(x, weight, bias)
    
    @property
    def weight(self):
        """Access weight in PyTorch-like format."""
        return self.mlx_linear.weight
    
    @property
    def bias(self):
        """Access bias in PyTorch-like format."""
        return self.mlx_linear.bias
    
    def parameters(self):
        """Return parameters in PyTorch-like format."""
        params = []
        if self.mlx_linear.weight.size > 0:
            params.append(self.mlx_linear.weight)
        if self.mlx_linear.bias.size > 0:
            params.append(self.mlx_linear.bias)
        return params
    
    def to(self, device):
        """Device compatibility (no-op for MLX)."""
        return self
    
    def eval(self):
        """Evaluation mode (no-op for MLX)."""
        return self
    
    def train(self, mode=True):
        """Training mode (no-op for MLX)."""
        return self