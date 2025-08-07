from typing import List, NamedTuple, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from ._irreps import Irreps
from ._tensor_product._codegen import _sum_tensors
from e3nn_mlx.util import prod


class Instruction(NamedTuple):
    i_in: int
    i_out: int
    path_shape: tuple
    path_weight: float


class Linear(nn.Module):
    r"""Linear operation equivariant to :math:`O(3)`

    Notes
    -----
        `e3nn.o3.Linear` objects created with different partitionings of the same irreps, such as ``Linear("10x0e", "0e")``
        and ``Linear("3x0e + 7x0e", "0e")``, are *not* equivalent: the second module has more instructions, which affects
        normalization. In a rough sense:

            Linear("10x0e", "0e") = normalization_coeff_0 * W_0 @ input
            Linear("3x0e + 7x0e", "0e") = normalization_coeff_1 * W_1 @ input[:3] + normalization_coeff_2 * W_2 @ input[3:]

        To make them equivalent, simplify ``irreps_in`` before constructing network modules:

            o3.Irreps("3x0e + 7x0e").simplify()  # => 10x0e


    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        representation of the input

    irreps_out : `e3nn.o3.Irreps`
        representation of the output

    internal_weights : bool
        whether the `e3nn.o3.Linear` should store its own weights. Defaults to ``True`` unless ``shared_weights`` is
        explicitly set to ``False``, for consistancy with `e3nn.o3.TensorProduct`.

    shared_weights : bool
        whether the `e3nn.o3.Linear` should be weighted individually for each input in a batch. Defaults to ``True``.
        Cannot be ``False`` if ``internal_weights`` is ``True``.

    instructions : list of 2-tuples, optional
        list of tuples ``(i_in, i_out)`` indicating which irreps in ``irreps_in`` should contribute to which irreps in
        ``irreps_out``. If ``None`` (the default), all allowable instructions will be created: every ``(i_in, i_out)`` such
        that ``irreps_in[i_in].ir == irreps_out[i_out].ir``.

    biases : list of bool, optional
        indicates for each element of ``irreps_out`` if it has a bias. By default there is no bias.
        If ``biases=True`` it gives bias to all scalars (l=0 and p=1).

    Attributes
    ----------
    weight_numel : int
        the size of the weights for this `e3nn.o3.Linear`

    Examples
    --------
    Linearly combines 4 scalars into 8 scalars and 16 vectors into 8 vectors.

    >>> lin = Linear("4x0e+16x1o", "8x0e+8x1o")
    >>> lin.weight_numel
    160

    Create a "block sparse" linear that does not combine two different groups of scalars;
    note that the number of weights is 4*4 + 3*3 = 25:

    >>> lin = Linear("4x0e + 3x0e", "4x0e + 3x0e", instructions=[(0, 0), (1, 1)])
    >>> lin.weight_numel
    25

    Be careful: because they have different instructions, the following two operations are not normalized in the same way,
    even though they contain all the same "connections":

    >>> lin1 = Linear("10x0e", "0e")
    >>> lin2 = Linear("3x0e + 7x0e", "0e")
    >>> lin1.weight_numel == lin2.weight_numel
    True
    >>> with torch.no_grad():
    ...     lin1.weight.fill_(1.0)
    ...     lin2.weight.fill_(1.0)
    Parameter containing:
    ...
    >>> x = torch.arange(10.0)
    >>> (lin1(x) - lin2(x)).abs().item() < 1e-5
    True

    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        *,
        f_in: Optional[int] = None,
        f_out: Optional[int] = None,
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        instructions: Optional[List[Tuple[int, int]]] = None,
        biases: Union[bool, List[bool]] = False,
        path_normalization: str = "element",
    ) -> None:
        super().__init__()

        assert path_normalization in ["element", "path"]

        irreps_in = Irreps(irreps_in)
        irreps_out = Irreps(irreps_out)

        if instructions is None:
            # By default, make all possible connections
            instructions = [
                (i_in, i_out)
                for i_in, (_, ir_in) in enumerate(irreps_in)
                for i_out, (_, ir_out) in enumerate(irreps_out)
                if ir_in == ir_out
            ]

        instructions = [
            Instruction(
                i_in=i_in,
                i_out=i_out,
                path_shape=(irreps_in[i_in].mul, irreps_out[i_out].mul),
                path_weight=1,
            )
            for i_in, i_out in instructions
        ]

        def alpha(ins) -> float:
            x = sum(
                irreps_in[i.i_in if path_normalization == "element" else ins.i_in].mul
                for i in instructions
                if i.i_out == ins.i_out
            )
            if f_in is not None:
                x *= f_in
            return 1.0 if x == 0 else x

        instructions = [
            Instruction(i_in=ins.i_in, i_out=ins.i_out, path_shape=ins.path_shape, path_weight=alpha(ins) ** (-0.5))
            for ins in instructions
        ]

        for ins in instructions:
            if not ins.i_in < len(irreps_in):
                raise IndexError(f"{ins.i_in} is not a valid index for irreps_in")
            if not ins.i_out < len(irreps_out):
                raise IndexError(f"{ins.i_out} is not a valid index for irreps_out")
            if not (ins.i_in == -1 or irreps_in[ins.i_in].ir == irreps_out[ins.i_out].ir):
                raise ValueError(f"{ins.i_in} and {ins.i_out} do not have the same irrep")

        if biases is None:
            biases = len(irreps_out) * (False,)
        if isinstance(biases, bool):
            biases = [biases and ir.is_scalar() for _, ir in irreps_out]

        assert len(biases) == len(irreps_out)
        assert all(ir.is_scalar() or (not b) for b, (_, ir) in zip(biases, irreps_out))

        instructions += [
            Instruction(i_in=-1, i_out=i_out, path_shape=(mul_ir.dim,), path_weight=1.0)
            for i_out, (bias, mul_ir) in enumerate(zip(biases, irreps_out))
            if bias
        ]

        # == Process arguments ==
        if shared_weights is False and internal_weights is None:
            internal_weights = False

        if shared_weights is None:
            shared_weights = True

        if internal_weights is None:
            internal_weights = True

        assert shared_weights or not internal_weights
        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.instructions = instructions

        # == Generate code ==
        self.weight_numel, self.bias_numel = self._compute_weight_bias_numel()

        # == Generate weights ==
        if internal_weights and self.weight_numel > 0:
            assert self.shared_weights, "Having internal weights impose shared weights"
            shape = ((f_in, f_out) if f_in is not None else ()) + (self.weight_numel,)
            self.weight = mx.random.normal(shape)
        else:
            # For compatibility, create empty weight
            self.weight = mx.array([])

        # == Generate biases ==
        if internal_weights and self.bias_numel > 0:
            assert self.shared_weights, "Having internal weights impose shared weights"
            shape = ((f_out,) if f_out is not None else ()) + (self.bias_numel,)
            self.bias = mx.zeros(shape)
        else:
            self.bias = mx.array([])

        # == Compute output mask ==
        if self.irreps_out.dim > 0:
            output_mask = mx.concatenate(
                [
                    (
                        mx.ones(mul_ir.dim)
                        if any((ins.i_out == i_out) and (0 not in ins.path_shape) for ins in self.instructions)
                        else mx.zeros(mul_ir.dim)
                    )
                    for i_out, mul_ir in enumerate(self.irreps_out)
                ]
            )
        else:
            output_mask = mx.ones(0)
        self.output_mask = output_mask

    def _compute_weight_bias_numel(self) -> Tuple[int, int]:
        """Compute the number of weight and bias parameters."""
        weight_numel = sum(prod(ins.path_shape) for ins in self.instructions if ins.i_in != -1)
        bias_numel = sum(self.irreps_out[ins.i_out].dim for ins in self.instructions if ins.i_in == -1)
        return weight_numel, bias_numel

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.irreps_in} -> {self.irreps_out} | {self.weight_numel} weights)"

    def __call__(self, features, weight: Optional[mx.array] = None, bias: Optional[mx.array] = None):
        """evaluate

        Parameters
        ----------
        features : `mx.array`
            tensor of shape ``(..., irreps_in.dim)``

        weight : `mx.array`, optional
            required if ``internal_weights`` is `False`

        Returns
        -------
        `mx.array`
            tensor of shape ``(..., irreps_out.dim)``
        """
        if weight is None:
            if self.weight_numel > 0 and not self.internal_weights:
                raise RuntimeError("Weights must be provided when internal_weights = False")
            weight = self.weight
        if bias is None:
            if self.bias_numel > 0 and not self.internal_weights:
                raise RuntimeError("Biases must be provided when internal_weights = False")
            bias = self.bias

        return self._compiled_main(features, weight, bias)

    def _compiled_main(self, x, weights, biases):
        """Main computation method for MLX Linear."""
        batch_shape = x.shape[:-1]
        
        # Handle empty input case
        if x.shape[-1] == 0:
            # Return zeros with the correct output shape
            return mx.zeros(batch_shape + (self.irreps_out.dim,), dtype=x.dtype)
        
        x_reshaped = x.reshape(-1, x.shape[-1])
        
        # Extract input irreps
        input_list = []
        start = 0
        for mul_ir in self.irreps_in:
            end = start + mul_ir.dim
            if mul_ir.dim > 0:
                input_list.append(x_reshaped[..., start:end].reshape(-1, mul_ir.mul, mul_ir.ir.dim))
            else:
                # Handle empty irreps (zero multiplicity) - use shape that works with einsum
                input_list.append(mx.zeros((x_reshaped.shape[0], mul_ir.mul if mul_ir.mul > 0 else 1, mul_ir.ir.dim if mul_ir.ir.dim > 0 else 1)))
            start = end

        # Process weights
        if self.weight_numel > 0 and weights.size > 0:
            weight_list = []
            start = 0
            for ins in self.instructions:
                if ins.i_in != -1:
                    end = start + prod(ins.path_shape)
                    weight_slice = weights[..., start:end]
                    if weight_slice.size > 0:
                        weight = weight_slice.reshape(-1, *ins.path_shape)
                    else:
                        # Handle empty weight slices - ensure proper shape for einsum
                        if ins.path_shape[0] == 0 or ins.path_shape[1] == 0:
                            # Handle zero multiplicity cases
                            weight = mx.zeros((1, max(1, ins.path_shape[0]), max(1, ins.path_shape[1])))
                        else:
                            weight = mx.zeros((1,) + ins.path_shape)
                    weight_list.append((ins, weight))
                    start = end
        else:
            weight_list = []

        # Process biases
        bias_list = []
        if self.bias_numel > 0 and biases.size > 0:
            start = 0
            for ins in self.instructions:
                if ins.i_in == -1:
                    end = start + prod(ins.path_shape)
                    bias = biases[..., start:end]
                    bias_list.append((ins, bias))
                    start = end

        # Compute outputs for each output irrep
        outputs = []
        for i_out, mul_ir_out in enumerate(self.irreps_out):
            output_parts = []
            
            # Add contributions from inputs
            for ins, weight in weight_list:
                if ins.i_out == i_out:
                    input_idx = ins.i_in
                    if input_idx >= 0 and input_idx < len(input_list):
                        # Apply linear transformation
                        weight_scaled = weight * ins.path_weight
                        # Handle broadcasting for higher dimensional weights
                        if weight_scaled.ndim > 2:
                            # Handle weights with additional dimensions (f_in, f_out, etc.)
                            # Use einsum with proper broadcasting
                            if weight_scaled.ndim == 3:
                                # Shape: (batch_or_f, u, w)
                                transformed = mx.einsum('zui,zuw->zwi', input_list[input_idx], weight_scaled)
                            elif weight_scaled.ndim == 4:
                                # Shape: (batch_or_f, f_in/f_out, u, w)
                                transformed = mx.einsum('zui,zfuw->zfwi', input_list[input_idx], weight_scaled)
                            else:
                                # Fallback to reshape and matmul
                                orig_shape = input_list[input_idx].shape
                                input_flat = input_list[input_idx].reshape(-1, orig_shape[-1])
                                weight_flat = weight_scaled.reshape(-1, weight_scaled.shape[-1])
                                transformed_flat = mx.matmul(input_flat, weight_flat.T)
                                transformed = transformed_flat.reshape(orig_shape[:-1] + (weight_flat.shape[0],))
                        else:
                            # Correct einsum for linear transformation:
                            # input: (batch, mul_in, irrep_dim)
                            # weight: (batch, mul_in, mul_out) 
                            # output: (batch, mul_out, irrep_dim)
                            transformed = mx.einsum('...ui,...uw->...wi', input_list[input_idx], weight_scaled)
                        # Handle reshape for zero multiplicity
                        if mul_ir_out.mul > 0 and mul_ir_out.ir.dim > 0:
                            transformed = transformed.reshape(-1, mul_ir_out.mul * mul_ir_out.ir.dim)
                        else:
                            transformed = mx.zeros((transformed.shape[0], 0))
                        output_parts.append(transformed)
            
            # Add bias contributions
            for ins, bias in bias_list:
                if ins.i_out == i_out:
                    bias_expanded = mx.broadcast_to(bias, (x_reshaped.shape[0], prod(ins.path_shape)))
                    output_parts.append(bias_expanded * ins.path_weight)
            
            if output_parts:
                # Filter out empty tensors before summing
                non_empty_parts = [part for part in output_parts if part.size > 0]
                if non_empty_parts:
                    output = sum(non_empty_parts)
                else:
                    output = mx.zeros((x_reshaped.shape[0], mul_ir_out.dim if mul_ir_out.dim > 0 else 0))
            else:
                output = mx.zeros((x_reshaped.shape[0], mul_ir_out.dim if mul_ir_out.dim > 0 else 0))
            
            outputs.append(output)

        # Concatenate all outputs
        # Filter out empty outputs
        non_empty_outputs = [out for out in outputs if out.size > 0]
        if non_empty_outputs:
            result = mx.concatenate(non_empty_outputs, axis=-1)
        else:
            result = mx.zeros((x_reshaped.shape[0], self.irreps_out.dim if self.irreps_out.dim > 0 else 0))
            
        return result.reshape(*batch_shape, self.irreps_out.dim if self.irreps_out.dim > 0 else 0)

    def weight_view_for_instruction(self, instruction: int, weight: Optional[mx.array] = None) -> mx.array:
        r"""View of weights corresponding to ``instruction``.

        Parameters
        ----------
        instruction : int
            The index of the instruction to get a view on the weights for.

        weight : `mx.array`, optional
            like ``weight`` argument to ``forward()``

        Returns
        -------
        `mx.array`
            A view on ``weight`` or this object's internal weights for the weights corresponding to the ``instruction`` th
            instruction.
        """
        if weight is None:
            assert self.internal_weights, "Weights must be provided when internal_weights = False"
            weight = self.weight
        
        offset = sum(prod(ins.path_shape) for ins in self.instructions[:instruction])
        ins = self.instructions[instruction]
        return weight[..., offset:offset + prod(ins.path_shape)].reshape(weight.shape[:-1] + ins.path_shape)

    def weight_views(self, weight: Optional[mx.array] = None, yield_instruction: bool = False):
        r"""Iterator over weight views for all instructions.

        Parameters
        ----------
        weight : `mx.array`, optional
            like ``weight`` argument to ``forward()``

        yield_instruction : `bool`, default False
            Whether to also yield the corresponding instruction.

        Yields
        ------
        If ``yield_instruction`` is ``True``, yields ``(instruction_index, instruction, weight_view)``.
        Otherwise, yields ``weight_view``.
        """
        if weight is None:
            assert self.internal_weights, "Weights must be provided when internal_weights = False"
            weight = self.weight
        
        offset = 0
        for ins_i, ins in enumerate(self.instructions):
            flatsize = prod(ins.path_shape)
            this_weight = weight[..., offset:offset + flatsize].reshape(weight.shape[:-1] + ins.path_shape)
            offset += flatsize
            if yield_instruction:
                yield ins_i, ins, this_weight
            else:
                yield this_weight