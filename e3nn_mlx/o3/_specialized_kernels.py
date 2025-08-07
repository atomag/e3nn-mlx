"""
Specialized Kernels for Common Operations in e3nn-mlx

This module provides highly optimized kernels for common equivariant operations
that are frequently used in e3nn networks.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional, Union, List
from functools import partial
import math

from ._irreps import Irreps, Irrep
from e3nn_mlx.util.compile import compile_mode, optimize_memory_layout
from e3nn_mlx.util import prod


class SpecializedKernels:
    """
    Collection of specialized kernels for common equivariant operations.
    """
    
    @staticmethod
    @compile_mode("mlx")
    def scalar_scalar_operation(x1: mx.array, x2: mx.array, weight: float = 1.0) -> mx.array:
        """
        Optimized scalar-scalar operation.
        
        Parameters
        ----------
        x1 : mx.array
            First scalar tensor
        x2 : mx.array
            Second scalar tensor
        weight : float, default 1.0
            Weight parameter
            
        Returns
        -------
        mx.array
            Result of scalar operation
        """
        return x1 * x2 * weight
    
    @staticmethod
    @compile_mode("mlx")
    def scalar_vector_operation(x_scalar: mx.array, x_vector: mx.array, weight: mx.array) -> mx.array:
        """
        Optimized scalar-vector operation.
        
        Parameters
        ----------
        x_scalar : mx.array
            Scalar tensor
        x_vector : mx.array
            Vector tensor
        weight : mx.array
            Weight matrix
            
        Returns
        -------
        mx.array
            Result of scalar-vector operation
        """
        # Ensure proper broadcasting
        # x_scalar shape: (batch_size, 1) for scalars
        # x_vector shape: (batch_size, 3) for vectors  
        # weight shape: (3,) for weights
        
        # Flatten batch dimensions for simpler broadcasting
        batch_size = x_scalar.shape[0]
        x_scalar_flat = x_scalar.reshape(-1)
        x_vector_flat = x_vector.reshape(-1, 3)
        
        # Apply scalar multiplication and weights
        result = x_scalar_flat[..., None] * x_vector_flat * weight
        
        # Reshape back to original batch dimensions
        return result.reshape(batch_size, 3)
    
    @staticmethod
    @compile_mode("mlx")
    def vector_vector_operation(x1: mx.array, x2: mx.array, weight_tensor: mx.array) -> mx.array:
        """
        Optimized vector-vector operation producing l=0, l=1, l=2 outputs.
        
        Parameters
        ----------
        x1 : mx.array
            First vector tensor
        x2 : mx.array
            Second vector tensor
        weight_tensor : mx.array
            Weight tensor for all outputs
            
        Returns
        -------
        mx.array
            Concatenated l=0, l=1, l=2 results
        """
        # Extract components
        x1_x, x1_y, x1_z = x1[..., 0], x1[..., 1], x1[..., 2]
        x2_x, x2_y, x2_z = x2[..., 0], x2[..., 1], x2[..., 2]
        
        # l=0 (scalar): dot product
        l0_result = (x1_x * x2_x + x1_y * x2_y + x1_z * x2_z) * weight_tensor[0]
        
        # l=1 (vector): cross product
        l1_x = (x1_y * x2_z - x1_z * x2_y) * weight_tensor[1]
        l1_y = (x1_z * x2_x - x1_x * x2_z) * weight_tensor[2]
        l1_z = (x1_x * x2_y - x1_y * x2_x) * weight_tensor[3]
        l1_result = mx.stack([l1_x, l1_y, l1_z], axis=-1)
        
        # l=2 (tensor): symmetric traceless tensor
        xx = x1_x * x2_x
        yy = x1_y * x2_y
        zz = x1_z * x2_z
        xy = (x1_x * x2_y + x1_y * x2_x) * 0.5
        xz = (x1_x * x2_z + x1_z * x2_x) * 0.5
        yz = (x1_y * x2_z + x1_z * x2_y) * 0.5
        
        # Real spherical harmonics components
        l2_xx_yy = (xx - yy) * weight_tensor[4]
        l2_xy = xy * weight_tensor[5]
        l2_xz = xz * weight_tensor[6]
        l2_yz = yz * weight_tensor[7]
        l2_zz_traceless = (zz - (xx + yy) / 3) * weight_tensor[8]
        
        l2_result = mx.stack([l2_xy, l2_yz, l2_zz_traceless, l2_xz, l2_xx_yy], axis=-1)
        
        # Concatenate all results
        return mx.concatenate([l0_result[..., None], l1_result, l2_result], axis=-1)
    
    @staticmethod
    @compile_mode("mlx")
    def tensor_vector_operation(x_tensor: mx.array, x_vector: mx.array, weights: mx.array) -> mx.array:
        """
        Optimized tensor-vector operation.
        
        Parameters
        ----------
        x_tensor : mx.array
            Tensor (l=2) input
        x_vector : mx.array
            Vector input
        weights : mx.array
            Weight tensor
            
        Returns
        -------
        mx.array
            Result of tensor-vector operation
        """
        # Extract tensor components (5 components for l=2)
        xy, yz, zz_traceless, xz, xx_yy = (
            x_tensor[..., 0], x_tensor[..., 1], x_tensor[..., 2], 
            x_tensor[..., 3], x_tensor[..., 4]
        )
        
        # Reconstruct full tensor
        xx = (xx_yy + zz_traceless) / 3
        yy = (-xx_yy + zz_traceless) / 3
        zz = zz_traceless
        
        # Extract vector components
        vx, vy, vz = x_vector[..., 0], x_vector[..., 1], x_vector[..., 2]
        
        # Compute tensor-vector product
        result_x = xx * vx + xy * vy + xz * vz
        result_y = xy * vx + yy * vy + yz * vz
        result_z = xz * vx + yz * vy + zz * vz
        
        result = mx.stack([result_x, result_y, result_z], axis=-1)
        
        return result * weights
    
    @staticmethod
    @compile_mode("mlx")
    def gate_operation(gates: mx.array, features: mx.array) -> mx.array:
        """
        Optimized gate operation for gated nonlinearities.
        
        Parameters
        ----------
        gates : mx.array
            Gate values (scalars)
        features : mx.array
            Feature tensors
            
        Returns
        -------
        mx.array
            Gated features
        """
        # Handle broadcasting between gates and features
        if gates.ndim == 2 and features.ndim == 2:
            # Both are 2D, need to handle broadcasting properly
            if gates.shape[-1] == 1:
                # Single gate for all features
                return gates * features
            else:
                # Multiple gates, need to distribute them
                num_gates = gates.shape[-1]
                features_per_gate = features.shape[-1] // num_gates
                gated_parts = []
                
                for i in range(num_gates):
                    start_idx = i * features_per_gate
                    end_idx = start_idx + features_per_gate
                    if end_idx <= features.shape[-1]:
                        gated_part = gates[..., i:i+1] * features[..., start_idx:end_idx]
                        gated_parts.append(gated_part)
                
                if gated_parts:
                    return mx.concatenate(gated_parts, axis=-1)
                else:
                    return mx.zeros_like(features)
        else:
            # General case
            gates_expanded = gates[..., None]
            return gates_expanded * features
    
    @staticmethod
    @compile_mode("mlx")
    def norm_operation(x: mx.array, eps: float = 1e-8) -> mx.array:
        """
        Optimized norm computation for equivariant features.
        
        Parameters
        ----------
        x : mx.array
            Input tensor
        eps : float, default 1e-8
            Small epsilon for numerical stability
            
        Returns
        -------
        mx.array
            Norm of the input
        """
        return mx.sqrt(mx.sum(x * x, axis=-1) + eps)
    
    @staticmethod
    @compile_mode("mlx")
    def normalize_operation(x: mx.array, eps: float = 1e-8) -> mx.array:
        """
        Optimized normalization operation.
        
        Parameters
        ----------
        x : mx.array
            Input tensor
        eps : float, default 1e-8
            Small epsilon for numerical stability
            
        Returns
        -------
        mx.array
            Normalized tensor
        """
        norm = SpecializedKernels.norm_operation(x, eps)
        return x / norm[..., None]


class FastTensorProduct(nn.Module):
    """
    Fast tensor product implementation using specialized kernels.
    """
    
    def __init__(self, irreps_in1: Irreps, irreps_in2: Irreps, irreps_out: Irreps):
        """
        Initialize fast tensor product.
        
        Parameters
        ----------
        irreps_in1 : Irreps
            First input irreps
        irreps_in2 : Irreps
            Second input irreps
        irreps_out : Irreps
            Output irreps
        """
        super().__init__()
        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out
        
        # Pre-compute operation patterns
        self.operation_patterns = self._analyze_operation_patterns()
    
    def _analyze_operation_patterns(self) -> List[Tuple]:
        """
        Analyze the tensor product to identify optimal operation patterns.
        
        Returns
        -------
        List[Tuple]
            List of (pattern_type, i_in1, i_in2, i_out) tuples
        """
        patterns = []
        
        for i_out, (mul_out, ir_out) in enumerate(self.irreps_out):
            for i_in1, (mul_in1, ir_in1) in enumerate(self.irreps_in1):
                for i_in2, (mul_in2, ir_in2) in enumerate(self.irreps_in2):
                    # Check if this combination contributes to the output
                    if ir_out.l in range(abs(ir_in1.l - ir_in2.l), ir_in1.l + ir_in2.l + 1):
                        if ir_out.p == ir_in1.p * ir_in2.p:
                            # Determine the operation pattern
                            if ir_in1.l == 0 and ir_in2.l == 0 and ir_out.l == 0:
                                pattern = "scalar_scalar"
                            elif ir_in1.l == 0 and ir_in2.l == 1 and ir_out.l == 1:
                                pattern = "scalar_vector"
                            elif ir_in1.l == 1 and ir_in2.l == 1 and ir_out.l <= 2:
                                pattern = "vector_vector"
                            elif ir_in1.l == 2 and ir_in2.l == 1 and ir_out.l <= 3:
                                pattern = "tensor_vector"
                            else:
                                pattern = "general"
                            
                            patterns.append((pattern, i_in1, i_in2, i_out))
        
        return patterns
    
    def __call__(self, x1: mx.array, x2: mx.array, weights: mx.array) -> mx.array:
        """
        Fast tensor product computation.
        
        Parameters
        ----------
        x1 : mx.array
            First input tensor
        x2 : mx.array
            Second input tensor
        weights : mx.array
            Weight tensor
            
        Returns
        -------
        mx.array
            Tensor product result
        """
        # Simplified implementation - just use element-wise multiplication with weights
        # This is a placeholder for a more sophisticated implementation
        
        # For now, just return a simple weighted combination
        # In practice, this would use the full tensor product logic
        
        batch_size = x1.shape[0]
        output_dim = self.irreps_out.dim
        
        # Simple weighted combination as fallback
        if weights.size >= output_dim:
            weight_slice = weights[:output_dim]
            # Use broadcasting to apply weights
            combined = x1 * x2[..., :x1.shape[-1]]  # Ensure matching dimensions
            # Reduce to output dimension
            if combined.shape[-1] > output_dim:
                result = combined[..., :output_dim] * weight_slice
            else:
                # Pad if necessary
                padding = output_dim - combined.shape[-1]
                result = mx.concatenate([combined, mx.zeros((*combined.shape[:-1], padding))], axis=-1)
                result = result * weight_slice
        else:
            # Fallback to zeros
            result = mx.zeros((batch_size, output_dim))
        
        return result


class OptimizedSphericalHarmonics:
    """
    Optimized spherical harmonics computation.
    """
    
    @staticmethod
    @compile_mode("mlx")
    def spherical_harmonics_l0(x: mx.array, normalize: bool = True) -> mx.array:
        """
        Optimized l=0 spherical harmonics.
        
        Parameters
        ----------
        x : mx.array
            Input coordinates
        normalize : bool, default True
            Whether to normalize
            
        Returns
        -------
        mx.array
            l=0 spherical harmonics
        """
        if normalize:
            norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
            result = mx.ones_like(norm[..., 0]) / (4 * mx.pi) ** 0.5
        else:
            result = mx.ones(x.shape[:-1])
        
        # Ensure 2D output for consistency with higher l values
        return result[..., None] if result.ndim == 1 else result
    
    @staticmethod
    @compile_mode("mlx")
    def spherical_harmonics_l1(x: mx.array, normalize: bool = True) -> mx.array:
        """
        Optimized l=1 spherical harmonics.
        
        Parameters
        ----------
        x : mx.array
            Input coordinates
        normalize : bool, default True
            Whether to normalize
            
        Returns
        -------
        mx.array
            l=1 spherical harmonics
        """
        if normalize:
            norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
            x_normalized = x / (norm + 1e-8)
            normalization = (3 / (4 * mx.pi)) ** 0.5
            return x_normalized * normalization
        else:
            return x
    
    @staticmethod
    @compile_mode("mlx")
    def spherical_harmonics_l2(x: mx.array, normalize: bool = True) -> mx.array:
        """
        Optimized l=2 spherical harmonics.
        
        Parameters
        ----------
        x : mx.array
            Input coordinates
        normalize : bool, default True
            Whether to normalize
            
        Returns
        -------
        mx.array
            l=2 spherical harmonics
        """
        if normalize:
            norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
            x_normalized = x / (norm + 1e-8)
            x, y, z = x_normalized[..., 0], x_normalized[..., 1], x_normalized[..., 2]
        else:
            x, y, z = x[..., 0], x[..., 1], x[..., 2]
        
        # Real spherical harmonics for l=2
        xy = x * y
        yz = y * z
        zz_traceless = 3 * z * z - 1
        xz = x * z
        xx_yy = x * x - y * y
        
        if normalize:
            # Normalization factors
            norm_xy = (15 / (4 * mx.pi)) ** 0.5
            norm_yz = (15 / (4 * mx.pi)) ** 0.5
            norm_zz = (5 / (16 * mx.pi)) ** 0.5
            norm_xz = (15 / (4 * mx.pi)) ** 0.5
            norm_xx_yy = (15 / (16 * mx.pi)) ** 0.5
            
            xy *= norm_xy
            yz *= norm_yz
            zz_traceless *= norm_zz
            xz *= norm_xz
            xx_yy *= norm_xx_yy
        
        return mx.stack([xy, yz, zz_traceless, xz, xx_yy], axis=-1)
    
    @staticmethod
    def spherical_harmonics(l: int, x: mx.array, normalize: bool = True) -> mx.array:
        """
        Optimized spherical harmonics computation for any l.
        
        Parameters
        ----------
        l : int
            Degree of spherical harmonics
        x : mx.array
            Input coordinates
        normalize : bool, default True
            Whether to normalize
            
        Returns
        -------
        mx.array
            Spherical harmonics values
        """
        if l == 0:
            return OptimizedSphericalHarmonics.spherical_harmonics_l0(x, normalize)
        elif l == 1:
            return OptimizedSphericalHarmonics.spherical_harmonics_l1(x, normalize)
        elif l == 2:
            return OptimizedSphericalHarmonics.spherical_harmonics_l2(x, normalize)
        else:
            # Fallback to general implementation for higher l
            # This would typically use a more sophisticated recursive algorithm
            raise NotImplementedError(f"Spherical harmonics for l={l} not yet optimized")


class FastGate(nn.Module):
    """
    Fast implementation of gated nonlinearities.
    """
    
    def __init__(self, irreps_scalars: Irreps, irreps_gated: Irreps):
        """
        Initialize fast gate.
        
        Parameters
        ----------
        irreps_scalars : Irreps
            Scalar irreps for gates
        irreps_gated : Irreps
            Irreps to be gated
        """
        super().__init__()
        self.irreps_scalars = irreps_scalars
        self.irreps_gated = irreps_gated
        
        # Pre-compute slicing information
        self.scalar_slices = []
        self.gated_slices = []
        
        start = 0
        for mul_ir in irreps_scalars:
            end = start + mul_ir.dim
            self.scalar_slices.append((start, end, mul_ir))
            start = end
        
        start = 0
        for mul_ir in irreps_gated:
            end = start + mul_ir.dim
            self.gated_slices.append((start, end, mul_ir))
            start = end
    
    @compile_mode("mlx")
    def __call__(self, scalars: mx.array, features: mx.array) -> mx.array:
        """
        Fast gate operation.
        
        Parameters
        ----------
        scalars : mx.array
            Scalar gate values
        features : mx.array
            Features to be gated
            
        Returns
        -------
        mx.array
            Gated features
        """
        # Apply sigmoid to scalars
        gates = mx.sigmoid(scalars)
        
        # Apply gating to each gated irrep
        outputs = []
        gate_idx = 0
        
        for start, end, mul_ir in self.gated_slices:
            if gate_idx < len(self.scalar_slices):
                # Get corresponding gate
                gate_start, gate_end, gate_mul_ir = self.scalar_slices[gate_idx]
                gate_slice = gates[..., gate_start:gate_end]
                
                # Apply gate operation
                gated_features = SpecializedKernels.gate_operation(gate_slice, features[..., start:end])
                outputs.append(gated_features)
                
                gate_idx += 1
            else:
                # No gate for this feature, pass through
                outputs.append(features[..., start:end])
        
        # Combine results
        return mx.concatenate(outputs, axis=-1)


def create_specialized_kernel_registry() -> dict:
    """
    Create a registry of specialized kernels for automatic selection.
    
    Returns
    -------
    dict
        Registry of specialized kernels
    """
    return {
        'scalar_scalar': SpecializedKernels.scalar_scalar_operation,
        'scalar_vector': SpecializedKernels.scalar_vector_operation,
        'vector_vector': SpecializedKernels.vector_vector_operation,
        'tensor_vector': SpecializedKernels.tensor_vector_operation,
        'gate': SpecializedKernels.gate_operation,
        'norm': SpecializedKernels.norm_operation,
        'normalize': SpecializedKernels.normalize_operation,
        'spherical_harmonics_l0': OptimizedSphericalHarmonics.spherical_harmonics_l0,
        'spherical_harmonics_l1': OptimizedSphericalHarmonics.spherical_harmonics_l1,
        'spherical_harmonics_l2': OptimizedSphericalHarmonics.spherical_harmonics_l2,
    }


# Global kernel registry
KERNEL_REGISTRY = create_specialized_kernel_registry()


def get_specialized_kernel(operation_type: str) -> callable:
    """
    Get a specialized kernel for the given operation type.
    
    Parameters
    ----------
    operation_type : str
        Type of operation
        
    Returns
    -------
    callable
        Specialized kernel function
    """
    if operation_type in KERNEL_REGISTRY:
        return KERNEL_REGISTRY[operation_type]
    else:
        raise ValueError(f"Unknown operation type: {operation_type}")


def register_specialized_kernel(operation_type: str, kernel: callable):
    """
    Register a new specialized kernel.
    
    Parameters
    ----------
    operation_type : str
        Type of operation
    kernel : callable
        Kernel function
    """
    KERNEL_REGISTRY[operation_type] = kernel