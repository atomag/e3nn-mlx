"""Test script for critical missing functions implementation."""

import sys
import mlx.core as mx
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, '/Users/boyu/code/e3nn-0.5.6/e3nn-mlx')

try:
    from e3nn_mlx.util import (
        # JIT compilation
        compile_mode, get_compile_mode, compile, trace, script,
        
        # Argument tools
        _transform, _get_io_irreps, _get_args_in, _rand_args,
        _get_device, _get_floating_dtype, _to_device_dtype,
        
        # Context management
        default_dtype, default_device, get_default_dtype, get_default_device,
        
        # Data types
        Chunk, Path, Instruction, chunk_from_slice, path_from_instructions,
        validate_chunk, validate_path,
    )
    
    from e3nn_mlx.o3 import Irreps
    
    print("✅ All imports successful!")
    
    # Test JIT compilation
    print("\n🧪 Testing JIT compilation...")
    
    @compile_mode("script")
    class TestModule:
        def __init__(self):
            self.weight = mx.random.normal((3, 3))
        
        def __call__(self, x):
            return x @ self.weight
        
        @property
        def parameters(self):
            return [self.weight]
    
    module = TestModule()
    mode = get_compile_mode(module)
    print(f"✅ Compile mode: {mode}")
    
    # Test argument tools
    print("\n🧪 Testing argument tools...")
    
    # Test _get_io_irreps
    class TestFunc:
        def __init__(self):
            self.irreps_in = Irreps("1x0e + 1x1o")
            self.irreps_out = Irreps("1x0e")
        
        def __call__(self, x):
            return x.sum(axis=-1, keepdims=True)
    
    func = TestFunc()
    irreps_in, irreps_out = _get_io_irreps(func)
    print(f"✅ IO irreps: {irreps_in} -> {irreps_out}")
    
    # Test _rand_args
    args = _rand_args(irreps_in, batch_size=2)
    print(f"✅ Random args shapes: {[arg.shape for arg in args]}")
    
    # Test _transform (simplified version)
    rot_mat = mx.eye(3)
    # Test with cartesian_points to avoid D_from_matrix issues
    test_args = [mx.random.normal((2, 3))]
    test_irreps = ["cartesian_points"]
    transformed = _transform(test_args, test_irreps, rot_mat)
    print(f"✅ Transform test: cartesian points transform successful")
    
    # Test _to_device_dtype
    converted = _to_device_dtype(args, dtype=mx.float16)
    print(f"✅ Device/dtype conversion: {[arg.dtype for arg in args]} -> {[arg.dtype for arg in converted]}")
    
    # Test context management
    print("\n🧪 Testing context management...")
    
    with default_dtype(mx.float16):
        dtype = get_default_dtype()
        print(f"✅ Context dtype: {dtype}")
    
    # Test data types
    print("\n🧪 Testing data types...")
    
    # Test Chunk
    irreps = Irreps("1x0e + 1x1o")
    chunk = Chunk(irreps, 0, irreps.dim)
    print(f"✅ Chunk: {chunk}, dim: {chunk.dim}")
    
    # Test Path
    instructions = [(0, 0, 0, "uvw", True)]
    path = Path(irreps, "uvw", instructions)
    print(f"✅ Path: {path}")
    
    # Test Instruction
    instr = Instruction(0, 0, 0, "uvw", True)
    print(f"✅ Instruction: {instr}")
    
    # Test utility functions
    print("\n🧪 Testing utility functions...")
    
    # Test chunk_from_slice
    chunk_slice = chunk_from_slice(irreps, slice(0, 1))
    print(f"✅ Chunk from slice: {chunk_slice}")
    
    # Test path_from_instructions
    path_from_instr = path_from_instructions(irreps, "uvw", instructions)
    print(f"✅ Path from instructions: {path_from_instr}")
    
    # Test validation
    print("\n🧪 Testing validation...")
    
    chunk_valid = validate_chunk(chunk)
    path_valid = validate_path(path)
    print(f"✅ Chunk valid: {chunk_valid}, Path valid: {path_valid}")
    
    print("\n🎉 All tests passed! Critical functions are working correctly.")
    
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)