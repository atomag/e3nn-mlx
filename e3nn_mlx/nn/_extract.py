from typing import Tuple
from e3nn_mlx.util._array_workarounds import array_at_set_workaround, spherical_harmonics_set_workaround


import mlx.core as mx
from mlx import nn

from ..o3._irreps import Irrep, Irreps


class Extract(nn.Module):
    # pylint: disable=abstract-method

    def __init__(self, irreps_in, irreps_outs, instructions, squeeze_out: bool = False) -> None:
        r"""Extract sub sets of irreps

        Parameters
        ----------
        irreps_in : `e3nn.o3.Irreps`
            representation of the input

        irreps_outs : list of `e3nn.o3.Irreps`
            list of representation of the outputs

        instructions : list of tuple of int
            list of tuples, one per output continaing each ``len(irreps_outs[i])`` int

        squeeze_out : bool, default False
            if ``squeeze_out`` and only one output exists, a ``mlx.core.array`` will be returned instead of a
            ``Tuple[mlx.core.array]``


        Examples
        --------

        >>> c = Extract('1e + 0e + 0e', ['0e', '0e'], [(1,), (2,)])
        >>> c(mx.array([0.0, 0.0, 0.0, 1.0, 2.0]))
        (mx.array([1.]), mx.array([2.]))
        """
        super().__init__()
        
        if not isinstance(irreps_in, (Irreps, str)):
            raise TypeError(f"irreps_in must be an Irreps object or string, got {type(irreps_in).__name__}")
        self.irreps_in = Irreps(irreps_in)
        
        if not isinstance(irreps_outs, (list, tuple)):
            raise TypeError(f"irreps_outs must be a list or tuple, got {type(irreps_outs).__name__}")
        self.irreps_outs = tuple(Irreps(irreps) for irreps in irreps_outs)
        
        if not isinstance(instructions, (list, tuple)):
            raise TypeError(f"instructions must be a list or tuple, got {type(instructions).__name__}")
        self.instructions = instructions
        
        if not isinstance(squeeze_out, bool):
            raise TypeError(f"squeeze_out must be a boolean, got {type(squeeze_out).__name__}")
        self.squeeze_out = squeeze_out

        if len(self.irreps_outs) != len(self.instructions):
            raise ValueError(
                f"Number of output irreps ({len(self.irreps_outs)}) must match "
                f"number of instruction sets ({len(self.instructions)})"
            )
        for i, (irreps_out, ins) in enumerate(zip(self.irreps_outs, self.instructions)):
            if len(irreps_out) != len(ins):
                raise ValueError(
                    f"Output irreps {i} has {len(irreps_out)} irreps but "
                    f"instruction set {i} has {len(ins)} instructions"
                )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(irreps_in={self.irreps_in}, irreps_outs={self.irreps_outs})"

    def __call__(self, x: mx.array):
        """evaluate"""
        if not isinstance(x, mx.array):
            raise TypeError(f"input must be an mx.array, got {type(x).__name__}")
        
        if x.ndim < 1:
            raise ValueError(f"input must have at least 1 dimension, got {x.ndim} dimensions")
        
        if x.shape[-1] != self.irreps_in.dim:
            raise ValueError(
                f"input last dimension ({x.shape[-1]}) must match irreps_in dimension ({self.irreps_in.dim})"
            )

        outputs = []
        for irreps_out, ins in zip(self.irreps_outs, self.instructions):
            output = mx.zeros(x.shape[:-1] + (irreps_out.dim,))
            
            if ins == tuple(range(len(self.irreps_in))):
                # Copy all input
                output = mx.array(x)
            else:
                # Extract specific irreps
                output_start = 0
                for i_in in ins:
                    input_start = sum(ir.dim for ir in self.irreps_in[:i_in])
                    input_dim = self.irreps_in[i_in].dim
                    
                    output_end = output_start + input_dim
                    # Use vectorized operations instead of nested loops
                    if x.ndim == 1:
                        # 1D array case - use concatenation
                        actual_len = min(output_end - output_start, x.shape[0] - input_start)
                        if actual_len > 0:
                            input_slice = x[input_start:input_start + actual_len]
                            
                            # Reconstruct output using concatenation
                            if output_start > 0:
                                before_slice = output[:output_start]
                            else:
                                before_slice = mx.array([], dtype=output.dtype)
                            
                            if output_start + actual_len < output.shape[0]:
                                after_slice = output[output_start + actual_len:]
                            else:
                                after_slice = mx.array([], dtype=output.dtype)
                            
                            # Concatenate the parts
                            output = mx.concatenate([before_slice, input_slice, after_slice], axis=0)
                    else:
                        # 2D array case - use concatenation
                        actual_len = min(output_end - output_start, x.shape[-1] - input_start)
                        if actual_len > 0:
                            input_slice = x[:, input_start:input_start + actual_len]
                            
                            # Reconstruct output using concatenation
                            if output_start > 0:
                                before_slice = output[:, :output_start]
                            else:
                                before_slice = mx.array([], dtype=output.dtype).reshape(x.shape[0], 0)
                            
                            if output_start + actual_len < output.shape[1]:
                                after_slice = output[:, output_start + actual_len:]
                            else:
                                after_slice = mx.array([], dtype=output.dtype).reshape(x.shape[0], 0)
                            
                            # Concatenate the parts
                            output = mx.concatenate([before_slice, input_slice, after_slice], axis=1)
                    output_start = output_end
            
            outputs.append(output)

        if self.squeeze_out and len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


class ExtractIr(Extract):
    # pylint: disable=abstract-method

    def __init__(self, irreps_in, ir) -> None:
        r"""Extract ``ir`` from irreps

        Parameters
        ----------
        irreps_in : `e3nn.o3.Irreps`
            representation of the input

        ir : `e3nn.o3.Irrep`
            representation to extract
        """
        ir = Irrep(ir)
        irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps([mul_ir for mul_ir in irreps_in if mul_ir.ir == ir])
        instructions = [tuple(i for i, mul_ir in enumerate(irreps_in) if mul_ir.ir == ir)]

        super().__init__(irreps_in, [self.irreps_out], instructions, squeeze_out=True)