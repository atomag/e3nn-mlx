"""
Adaptive Precision and Batching for e3nn-mlx

This module provides adaptive precision management and dynamic batching
optimizations for e3nn-mlx operations.
"""

import mlx.core as mx
from typing import Union, Optional, Tuple, List, Dict, Any, Callable
import time
import warnings
import math

from e3nn_mlx.util.compile import compile_mode, optimize_memory_layout
from ..o3._irreps import Irreps


class AdaptivePrecisionManager:
    """
    Manager for adaptive precision optimization.
    """
    
    def __init__(self, 
                 default_precision: str = "float32",
                 enable_mixed_precision: bool = True,
                 precision_threshold: float = 1e-6):
        """
        Initialize adaptive precision manager.
        
        Parameters
        ----------
        default_precision : str, default "float32"
            Default precision for operations
        enable_mixed_precision : bool, default True
            Whether to enable mixed precision
        precision_threshold : float, default 1e-6
            Threshold for precision decisions
        """
        self.default_precision = default_precision
        self.enable_mixed_precision = enable_mixed_precision
        self.precision_threshold = precision_threshold
        
        # Precision mapping
        self.precision_map = {
            "float16": mx.float16,
            "float32": mx.float32,
            "float64": mx.float64,
        }
        
        # Performance tracking
        self.precision_stats = {}
    
    def get_optimal_precision(self, tensor: mx.array, operation: str) -> mx.Dtype:
        """
        Determine optimal precision for a tensor and operation.
        
        Parameters
        ----------
        tensor : mx.array
            Input tensor
        operation : str
            Operation type
            
        Returns
        -------
        mx.Dtype
            Optimal precision
        """
        if not self.enable_mixed_precision:
            return self.precision_map[self.default_precision]
        
        # Analyze tensor characteristics
        tensor_range = mx.max(mx.abs(tensor))
        tensor_mean = mx.mean(mx.abs(tensor))
        
        # Precision selection logic
        if tensor_range < 1e4 and operation in ["linear", "gate"]:
            # Use float16 for small ranges and simple operations
            return mx.float16
        elif tensor_range < 1e6 and operation in ["tensor_product", "spherical_harmonics"]:
            # Use float32 for medium ranges
            return mx.float32
        else:
            # Use float32/float64 for large ranges or complex operations
            return mx.float32
    
    def convert_precision(self, tensor: mx.array, precision: mx.Dtype) -> mx.array:
        """
        Convert tensor to specified precision.
        
        Parameters
        ----------
        tensor : mx.array
            Input tensor
        precision : mx.Dtype
            Target precision
            
        Returns
        -------
        mx.array
            Converted tensor
        """
        if tensor.dtype == precision:
            return tensor
        
        return tensor.astype(precision)
    
    def mixed_precision_operation(self, 
                                 operation: Callable,
                                 inputs: List[mx.array],
                                 operation_type: str) -> mx.array:
        """
        Execute operation with mixed precision.
        
        Parameters
        ----------
        operation : Callable
            Operation to execute
        inputs : List[mx.array]
            Input tensors
        operation_type : str
            Type of operation
            
        Returns
        -------
        mx.array
            Operation result
        """
        if not self.enable_mixed_precision:
            return operation(*inputs)
        
        # Convert inputs to optimal precisions
        converted_inputs = []
        for tensor in inputs:
            precision = self.get_optimal_precision(tensor, operation_type)
            converted_inputs.append(self.convert_precision(tensor, precision))
        
        # Execute operation
        result = operation(*converted_inputs)
        
        # Convert result back to default precision
        return self.convert_precision(result, self.precision_map[self.default_precision])
    
    def track_precision_performance(self, 
                                   operation: str,
                                   precision: str,
                                   execution_time: float,
                                   error_metric: float):
        """
        Track performance of different precision levels.
        
        Parameters
        ----------
        operation : str
            Operation type
        precision : str
            Precision used
        execution_time : float
            Execution time
        error_metric : float
            Error metric compared to reference
        """
        if operation not in self.precision_stats:
            self.precision_stats[operation] = {}
        
        if precision not in self.precision_stats[operation]:
            self.precision_stats[operation][precision] = []
        
        self.precision_stats[operation][precision].append({
            'time': execution_time,
            'error': error_metric
        })
    
    def get_precision_recommendations(self, operation: str) -> Dict[str, float]:
        """
        Get precision recommendations for an operation.
        
        Parameters
        ----------
        operation : str
            Operation type
            
        Returns
        -------
        Dict[str, float]
            Precision recommendations with scores
        """
        if operation not in self.precision_stats:
            return {self.default_precision: 1.0}
        
        recommendations = {}
        for precision, stats in self.precision_stats[operation].items():
            if not stats:
                continue
            
            avg_time = sum(s['time'] for s in stats) / len(stats)
            avg_error = sum(s['error'] for s in stats) / len(stats)
            
            # Score based on speed and accuracy
            speed_score = 1.0 / avg_time
            accuracy_score = 1.0 / (1.0 + avg_error)
            
            recommendations[precision] = speed_score * accuracy_score
        
        return recommendations


class AdaptiveBatchProcessor:
    """
    Processor for adaptive batch size optimization.
    """
    
    def __init__(self, 
                 min_batch_size: int = 1,
                 max_batch_size: int = 1024,
                 target_memory_usage: float = 0.8,
                 enable_dynamic_batching: bool = True):
        """
        Initialize adaptive batch processor.
        
        Parameters
        ----------
        min_batch_size : int, default 1
            Minimum batch size
        max_batch_size : int, default 1024
            Maximum batch size
        target_memory_usage : float, default 0.8
            Target memory usage ratio
        enable_dynamic_batching : bool, default True
            Whether to enable dynamic batching
        """
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_memory_usage = target_memory_usage
        self.enable_dynamic_batching = enable_dynamic_batching
        
        # Performance tracking
        self.batch_stats = {}
        
        # Memory tracking
        self.memory_history = []
    
    def estimate_memory_usage(self, 
                            batch_size: int, 
                            input_shape: Tuple[int, ...],
                            output_shape: Tuple[int, ...]) -> float:
        """
        Estimate memory usage for a batch size.
        
        Parameters
        ----------
        batch_size : int
            Batch size
        input_shape : Tuple[int, ...]
            Input shape (excluding batch)
        output_shape : Tuple[int, ...]
            Output shape (excluding batch)
            
        Returns
        -------
        float
            Estimated memory usage in bytes
        """
        # Simple memory estimation
        input_elements = batch_size * math.prod(input_shape)
        output_elements = batch_size * math.prod(output_shape)
        
        # Assume float32 (4 bytes per element)
        memory_bytes = (input_elements + output_elements) * 4
        
        # Add overhead for intermediate computations
        memory_bytes *= 1.5
        
        return memory_bytes
    
    def get_optimal_batch_size(self, 
                             input_shape: Tuple[int, ...],
                             output_shape: Tuple[int, ...],
                             available_memory: Optional[float] = None) -> int:
        """
        Get optimal batch size based on memory constraints.
        
        Parameters
        ----------
        input_shape : Tuple[int, ...]
            Input shape (excluding batch)
        output_shape : Tuple[int, ...]
            Output shape (excluding batch)
        available_memory : float, optional
            Available memory in bytes
            
        Returns
        -------
        int
            Optimal batch size
        """
        if not self.enable_dynamic_batching:
            return self.min_batch_size
        
        if available_memory is None:
            # Use a reasonable default
            available_memory = 2 * 1024**3  # 2GB
        
        # Binary search for optimal batch size
        left, right = self.min_batch_size, self.max_batch_size
        optimal_size = self.min_batch_size
        
        while left <= right:
            mid = (left + right) // 2
            memory_needed = self.estimate_memory_usage(mid, input_shape, output_shape)
            
            if memory_needed <= available_memory * self.target_memory_usage:
                optimal_size = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return optimal_size
    
    def process_adaptive_batch(self, 
                             operation: Callable,
                             inputs: List[mx.array],
                             output_shape: Tuple[int, ...]) -> mx.array:
        """
        Process inputs with adaptive batching.
        
        Parameters
        ----------
        operation : Callable
            Operation to execute
        inputs : List[mx.array]
            Input tensors
        output_shape : Tuple[int, ...]
            Expected output shape (excluding batch)
            
        Returns
        -------
        mx.array
            Operation result
        """
        if not self.enable_dynamic_batching:
            return operation(*inputs)
        
        batch_size = inputs[0].shape[0]
        input_shape = inputs[0].shape[1:]
        
        # Get optimal batch size
        optimal_batch_size = self.get_optimal_batch_size(input_shape, output_shape)
        
        if batch_size <= optimal_batch_size:
            # Process as single batch
            return operation(*inputs)
        
        # Process in chunks
        outputs = []
        for i in range(0, batch_size, optimal_batch_size):
            chunk_inputs = [x[i:i + optimal_batch_size] for x in inputs]
            chunk_output = operation(*chunk_inputs)
            outputs.append(chunk_output)
        
        # Concatenate results
        return mx.concatenate(outputs, axis=0)
    
    def track_batch_performance(self, 
                              operation: str,
                              batch_size: int,
                              execution_time: float,
                              memory_usage: float):
        """
        Track batch processing performance.
        
        Parameters
        ----------
        operation : str
            Operation type
        batch_size : int
            Batch size used
        execution_time : float
            Execution time
        memory_usage : float
            Memory usage
        """
        if operation not in self.batch_stats:
            self.batch_stats[operation] = []
        
        self.batch_stats[operation].append({
            'batch_size': batch_size,
            'time': execution_time,
            'memory': memory_usage
        })
    
    def get_batch_recommendations(self, operation: str) -> Dict[str, Any]:
        """
        Get batch size recommendations for an operation.
        
        Parameters
        ----------
        operation : str
            Operation type
            
        Returns
        -------
        Dict[str, Any]
            Batch size recommendations
        """
        if operation not in self.batch_stats:
            return {'recommended_batch_size': self.min_batch_size}
        
        stats = self.batch_stats[operation]
        
        # Find batch size with best time/throughput ratio
        best_batch_size = self.min_batch_size
        best_throughput = 0
        
        for stat in stats:
            batch_size = stat['batch_size']
            time_per_sample = stat['time'] / batch_size
            throughput = 1.0 / time_per_sample
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_batch_size = batch_size
        
        return {
            'recommended_batch_size': best_batch_size,
            'best_throughput': best_throughput,
            'stats': stats
        }


class AdaptiveOptimizer:
    """
    Combined adaptive precision and batching optimizer.
    """
    
    def __init__(self, 
                 precision_manager: Optional[AdaptivePrecisionManager] = None,
                 batch_processor: Optional[AdaptiveBatchProcessor] = None):
        """
        Initialize adaptive optimizer.
        
        Parameters
        ----------
        precision_manager : AdaptivePrecisionManager, optional
            Precision manager instance
        batch_processor : AdaptiveBatchProcessor, optional
            Batch processor instance
        """
        self.precision_manager = precision_manager or AdaptivePrecisionManager()
        self.batch_processor = batch_processor or AdaptiveBatchProcessor()
        
        # Optimization history
        self.optimization_history = []
    
    def optimize_operation(self, 
                          operation: Callable,
                          inputs: List[mx.array],
                          operation_type: str,
                          output_shape: Optional[Tuple[int, ...]] = None) -> mx.array:
        """
        Optimize operation execution with adaptive precision and batching.
        
        Parameters
        ----------
        operation : Callable
            Operation to optimize
        inputs : List[mx.array]
            Input tensors
        operation_type : str
            Type of operation
        output_shape : Tuple[int, ...], optional
            Expected output shape
            
        Returns
        -------
        mx.array
            Optimized operation result
        """
        start_time = time.time()
        
        # Apply precision optimization
        optimized_operation = lambda *args: self.precision_manager.mixed_precision_operation(
            operation, args, operation_type
        )
        
        # Apply batching optimization
        if output_shape is not None:
            result = self.batch_processor.process_adaptive_batch(
                optimized_operation, inputs, output_shape
            )
        else:
            result = optimized_operation(*inputs)
        
        # Track performance
        execution_time = time.time() - start_time
        self._track_optimization_performance(operation_type, execution_time, inputs, result)
        
        return result
    
    def _track_optimization_performance(self, 
                                       operation_type: str,
                                       execution_time: float,
                                       inputs: List[mx.array],
                                       output: mx.array):
        """
        Track optimization performance.
        
        Parameters
        ----------
        operation_type : str
            Type of operation
        execution_time : float
            Execution time
        inputs : List[mx.array]
            Input tensors
        output : mx.array
            Output tensor
        """
        # Estimate memory usage
        input_memory = sum(x.size * 4 for x in inputs)  # Assume float32
        output_memory = output.size * 4
        total_memory = input_memory + output_memory
        
        # Record optimization stats
        self.optimization_history.append({
            'operation': operation_type,
            'time': execution_time,
            'memory': total_memory,
            'input_shapes': [x.shape for x in inputs],
            'output_shape': output.shape
        })
        
        # Update individual component stats
        self.precision_manager.track_precision_performance(
            operation_type, "mixed", execution_time, 0.0
        )
        
        batch_size = inputs[0].shape[0]
        self.batch_processor.track_batch_performance(
            operation_type, batch_size, execution_time, total_memory
        )
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get optimization summary and recommendations.
        
        Returns
        -------
        Dict[str, Any]
            Optimization summary
        """
        if not self.optimization_history:
            return {"message": "No optimization history available"}
        
        # Calculate summary statistics
        total_operations = len(self.optimization_history)
        total_time = sum(h['time'] for h in self.optimization_history)
        total_memory = sum(h['memory'] for h in self.optimization_history)
        
        avg_time = total_time / total_operations
        avg_memory = total_memory / total_operations
        
        # Get recommendations from components
        precision_recommendations = {}
        batch_recommendations = {}
        
        for operation in set(h['operation'] for h in self.optimization_history):
            prec_rec = self.precision_manager.get_precision_recommendations(operation)
            batch_rec = self.batch_processor.get_batch_recommendations(operation)
            
            if prec_rec:
                precision_recommendations[operation] = prec_rec
            if batch_rec:
                batch_recommendations[operation] = batch_rec
        
        return {
            'total_operations': total_operations,
            'total_time': total_time,
            'total_memory': total_memory,
            'average_time_per_operation': avg_time,
            'average_memory_per_operation': avg_memory,
            'precision_recommendations': precision_recommendations,
            'batch_recommendations': batch_recommendations,
            'optimization_history': self.optimization_history[-10:]  # Last 10 operations
        }
    
    def reset_stats(self):
        """Reset all optimization statistics."""
        self.precision_manager.precision_stats.clear()
        self.batch_processor.batch_stats.clear()
        self.optimization_history.clear()


def create_adaptive_optimized_function(
    operation: Callable,
    operation_type: str,
    output_shape: Optional[Tuple[int, ...]] = None,
    optimizer: Optional[AdaptiveOptimizer] = None
) -> Callable:
    """
    Create an adaptive optimized version of a function.
    
    Parameters
    ----------
    operation : Callable
        Function to optimize
    operation_type : str
        Type of operation
    output_shape : Tuple[int, ...], optional
        Expected output shape
    optimizer : AdaptiveOptimizer, optional
        Optimizer instance
        
    Returns
    -------
    Callable
        Optimized function
    """
    if optimizer is None:
        optimizer = AdaptiveOptimizer()
    
    def optimized_function(*args, **kwargs):
        inputs = list(args)
        return optimizer.optimize_operation(operation, inputs, operation_type, output_shape)
    
    return optimized_function


# Global optimizer instance
_global_optimizer = None


def get_global_optimizer() -> AdaptiveOptimizer:
    """Get the global adaptive optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = AdaptiveOptimizer()
    return _global_optimizer


def set_global_optimizer(optimizer: AdaptiveOptimizer):
    """Set the global adaptive optimizer instance."""
    global _global_optimizer
    _global_optimizer = optimizer


def optimize_with_adaptive_precision(operation: Callable, operation_type: str) -> Callable:
    """
    Decorator for adaptive precision optimization.
    
    Parameters
    ----------
    operation : Callable
        Function to optimize
    operation_type : str
        Type of operation
        
    Returns
    -------
    Callable
        Optimized function
    """
    optimizer = get_global_optimizer()
    
    def optimized_function(*args, **kwargs):
        inputs = list(args)
        return optimizer.optimize_operation(operation, inputs, operation_type)
    
    return optimized_function


def optimize_with_adaptive_batching(operation: Callable, 
                                  operation_type: str,
                                  output_shape: Tuple[int, ...]) -> Callable:
    """
    Decorator for adaptive batching optimization.
    
    Parameters
    ----------
    operation : Callable
        Function to optimize
    operation_type : str
        Type of operation
    output_shape : Tuple[int, ...]
        Expected output shape
        
    Returns
    -------
    Callable
        Optimized function
    """
    optimizer = get_global_optimizer()
    
    def optimized_function(*args, **kwargs):
        inputs = list(args)
        return optimizer.optimize_operation(operation, inputs, operation_type, output_shape)
    
    return optimized_function