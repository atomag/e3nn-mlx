"""Data types for e3nn-mlx.

This module provides named tuples and data structures used throughout e3nn,
adapted from e3nn's datatypes.py for MLX.
"""
from typing import NamedTuple, Optional, Union, List, Any


# Local import to avoid circular imports
def _get_irreps_class():
    from e3nn_mlx.o3 import Irreps
    return Irreps


class Chunk(NamedTuple):
    """A chunk of irreps with start and stop indices.
    
    Parameters
    ----------
    irreps : Irreps
        The irreducible representations in this chunk
    start : int
        Starting index in the full representation
    stop : int
        Ending index in the full representation
    """
    irreps: Any  # Will be Irreps when imported
    start: int
    stop: int
    
    @property
    def dim(self) -> int:
        """Dimension of this chunk."""
        return self.stop - self.start
    
    def __repr__(self) -> str:
        return f"Chunk(irreps={self.irreps}, start={self.start}, stop={self.stop})"


class Path(NamedTuple):
    """A path in a tensor product operation.
    
    Parameters
    ----------
    irreps_in : Irreps
        Input irreducible representations
    operation : str
        The operation type (e.g., 'uvw', 'uvu', etc.)
    instructions : List[tuple]
        List of instruction tuples (i, j, k, mode, train)
    """
    irreps_in: Any  # Will be Irreps when imported
    operation: str
    instructions: List[tuple]
    
    def __repr__(self) -> str:
        return f"Path(irreps_in={self.irreps_in}, operation='{self.operation}', instructions={self.instructions})"


class TensorProductPath(NamedTuple):
    """A complete tensor product path specification.
    
    Parameters
    ----------
    irreps_in1 : Irreps
        First input irreducible representations
    irreps_in2 : Irreps
        Second input irreducible representations
    irreps_out : Irreps
        Output irreducible representations
    path : Path
        The path specification
    """
    irreps_in1: Any  # Will be Irreps when imported
    irreps_in2: Any  # Will be Irreps when imported
    irreps_out: Any  # Will be Irreps when imported
    path: Path
    
    def __repr__(self) -> str:
        return (f"TensorProductPath(irreps_in1={self.irreps_in1}, "
                f"irreps_in2={self.irreps_in2}, "
                f"irreps_out={self.irreps_out}, "
                f"path={self.path})")


class OptimizedOperation(NamedTuple):
    """An optimized operation specification.
    
    Parameters
    ----------
    name : str
        Name of the operation
    irreps_in : List[Irreps]
        Input irreducible representations
    irreps_out : Irreps
        Output irreducible representations
    code : str
        Generated code for the operation
    """
    name: str
    irreps_in: List[Any]  # Will be List[Irreps] when imported
    irreps_out: Any  # Will be Irreps when imported
    code: str
    
    def __repr__(self) -> str:
        return (f"OptimizedOperation(name='{self.name}', "
                f"irreps_in={self.irreps_in}, "
                f"irreps_out={self.irreps_out}, "
                f"code=...)")


class Instruction(NamedTuple):
    """A single instruction in a tensor product.
    
    Parameters
    ----------
    i_in : int
        Index in first input
    j_in : int
        Index in second input
    k_out : int
        Index in output
    connection_mode : str
        How to connect the irreps
    train : bool
        Whether this connection is trainable
    """
    i_in: int
    j_in: int
    k_out: int
    connection_mode: str
    train: bool
    
    def __repr__(self) -> str:
        return (f"Instruction(i_in={self.i_in}, "
                f"j_in={self.j_in}, "
                f"k_out={self.k_out}, "
                f"connection_mode='{self.connection_mode}', "
                f"train={self.train})")


def chunk_from_slice(irreps: Any, slice_obj: slice) -> Chunk:
    """Create a Chunk from a slice of irreps.
    
    Parameters
    ----------
    irreps : Irreps
        The full irreducible representations
    slice_obj : slice
        The slice to extract
        
    Returns
    -------
    Chunk
        The chunk representing the slice
    """
    # Convert slice to start/stop indices
    start = slice_obj.start if slice_obj.start is not None else 0
    stop = slice_obj.stop if slice_obj.stop is not None else len(irreps)
    
    # Get the irreps in this slice
    chunk_irreps = irreps[slice_obj]
    
    # Convert to actual dimension indices
    # Get dimensions of each irrep
    dims = [ir.dim for mul, ir in irreps]
    start_dim = sum(dims[:start]) if start > 0 else 0
    stop_dim = sum(dims[:stop])
    
    return Chunk(chunk_irreps, start_dim, stop_dim)


def path_from_instructions(irreps_in: Any, operation: str, instructions: List[tuple]) -> Path:
    """Create a Path from a list of instructions.
    
    Parameters
    ----------
    irreps_in : Irreps
        Input irreducible representations
    operation : str
        The operation type
    instructions : List[tuple]
        List of instruction tuples
        
    Returns
    -------
    Path
        The path specification
    """
    # Convert instruction tuples to Instruction objects if needed
    processed_instructions = []
    for instr in instructions:
        if len(instr) == 5:
            # Already in (i, j, k, mode, train) format
            processed_instructions.append(Instruction(*instr))
        else:
            # Assume it's already an Instruction or similar
            processed_instructions.append(instr)
    
    return Path(irreps_in, operation, processed_instructions)


def validate_chunk(chunk: Chunk) -> bool:
    """Validate that a chunk is well-formed.
    
    Parameters
    ----------
    chunk : Chunk
        The chunk to validate
        
    Returns
    -------
    bool
        True if valid, False otherwise
    """
    # Check that start <= stop
    if chunk.start > chunk.stop:
        return False
    
    # Check that the irreps dimension matches the chunk dimension
    if chunk.irreps.dim != chunk.dim:
        return False
    
    return True


def validate_path(path: Path) -> bool:
    """Validate that a path is well-formed.
    
    Parameters
    ----------
    path : Path
        The path to validate
        
    Returns
    -------
    bool
        True if valid, False otherwise
    """
    # Check that operation is valid
    valid_operations = ['uvw', 'uvu', 'uuw', 'uuu']
    if path.operation not in valid_operations:
        return False
    
    # Check that instructions are well-formed
    for instr in path.instructions:
        if isinstance(instr, Instruction):
            if instr.i_in < 0 or instr.j_in < 0 or instr.k_out < 0:
                return False
            if instr.connection_mode not in ['uvw', 'uvu', 'uuw', 'uuu']:
                return False
        else:
            # Assume it's a tuple in the old format
            if len(instr) != 5:
                return False
            if instr[0] < 0 or instr[1] < 0 or instr[2] < 0:
                return False
            if instr[3] not in ['uvw', 'uvu', 'uuw', 'uuu']:
                return False
    
    return True