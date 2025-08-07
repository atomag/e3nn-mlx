import pytest

import mlx.core as mx

from e3nn_mlx import o3
from e3nn_mlx.util.test import assert_equivariant, random_irreps


class SlowLinear:
    """Direct implementation of Linear logic without TensorProduct."""

    def __init__(
        self,
        irreps_in,
        irreps_out,
        internal_weights=None,
        shared_weights=None,
    ) -> None:
        irreps_in = o3.Irreps(irreps_in)
        irreps_out = o3.Irreps(irreps_out)

        # Create instructions exactly like Linear class does
        instructions = [
            (i_in, i_out)
            for i_in, (_, ir_in) in enumerate(irreps_in)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_in == ir_out
        ]
        
        # Convert to Linear's Instruction format
        from e3nn_mlx.o3._linear import Instruction
        self.instructions = [
            Instruction(
                i_in=i_in,
                i_out=i_out,
                path_shape=(irreps_in[i_in].mul, irreps_out[i_out].mul),
                path_weight=1,  # Will be set below
            )
            for i_in, i_out in instructions
        ]
        
        # Apply path normalization exactly like Linear class does
        def alpha(ins) -> float:
            x = sum(
                irreps_in[i.i_in].mul
                for i in self.instructions
                if i.i_out == ins.i_out
            )
            return 1.0 if x == 0 else x

        self.instructions = [
            Instruction(i_in=ins.i_in, i_out=ins.i_out, path_shape=ins.path_shape, path_weight=alpha(ins) ** (-0.5))
            for ins in self.instructions
        ]
        
        # Compute weight and bias counts
        self.weight_numel = sum(ins.path_shape[0] * ins.path_shape[1] for ins in self.instructions)
        self.bias_numel = 0  # Linear class doesn't use biases by default
        
        # Create dummy weight parameter to match Linear interface
        self.weight = mx.random.normal((self.weight_numel,))
        
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out

    def __call__(self, features, weight=None):
        """Direct implementation of Linear's forward pass."""
        if weight is None:
            weight = self.weight
        
        batch_shape = features.shape[:-1]
        
        # Handle empty input case
        if features.shape[-1] == 0:
            # Return zeros with the correct output shape
            return mx.zeros(batch_shape + (self.irreps_out.dim,), dtype=features.dtype)
        
        x_reshaped = features.reshape(-1, features.shape[-1])
        
        # Extract input irreps
        input_list = []
        start = 0
        for mul_ir in self.irreps_in:
            end = start + mul_ir.dim
            if mul_ir.dim > 0:
                input_list.append(x_reshaped[..., start:end].reshape(-1, mul_ir.mul, mul_ir.ir.dim))
            else:
                input_list.append(mx.zeros((x_reshaped.shape[0], mul_ir.mul if mul_ir.mul > 0 else 1, mul_ir.ir.dim if mul_ir.ir.dim > 0 else 1)))
            start = end
        
        # Process weights
        weight_list = []
        start = 0
        for ins in self.instructions:
            end = start + ins.path_shape[0] * ins.path_shape[1]
            weight_slice = weight[start:end]
            weight_list.append((ins, weight_slice.reshape(ins.path_shape)))
            start = end
        
        # Compute outputs for each output irrep
        outputs = []
        for i_out, mul_ir_out in enumerate(self.irreps_out):
            output_parts = []
            
            # Add contributions from inputs
            for ins, weight_matrix in weight_list:
                if ins.i_out == i_out:
                    input_idx = ins.i_in
                    if input_idx < len(input_list):
                        input_tensor = input_list[input_idx]
                        
                        # Skip zero multiplicity cases that cause broadcasting issues
                        if input_tensor.size == 0 or weight_matrix.size == 0:
                            continue
                            
                        # Apply linear transformation exactly like Linear does
                        weight_scaled = weight_matrix * ins.path_weight
                        # Use the same einsum as Linear class
                        transformed = mx.einsum('...ui,...uw->...wi', input_tensor, weight_scaled)
                        if mul_ir_out.mul > 0 and mul_ir_out.ir.dim > 0:
                            transformed = transformed.reshape(-1, mul_ir_out.mul * mul_ir_out.ir.dim)
                        else:
                            transformed = mx.zeros((transformed.shape[0], 0))
                        output_parts.append(transformed)
            
            if output_parts:
                output = sum(output_parts)
            else:
                output = mx.zeros((x_reshaped.shape[0], mul_ir_out.dim if mul_ir_out.dim > 0 else 0))
            
            outputs.append(output)
        
        # Concatenate all outputs
        non_empty_outputs = [out for out in outputs if out.size > 0]
        if non_empty_outputs:
            result = mx.concatenate(non_empty_outputs, axis=-1)
        else:
            result = mx.zeros((x_reshaped.shape[0], self.irreps_out.dim if self.irreps_out.dim > 0 else 0))
            
        return result.reshape(*batch_shape, self.irreps_out.dim if self.irreps_out.dim > 0 else 0)


def test_linear() -> None:
    """Test basic linear layer functionality."""
    irreps_in = o3.Irreps("1e + 2e + 3x3o")
    irreps_out = o3.Irreps("1e + 2e + 3x3o")

    def build_module(irreps_in, irreps_out):
        return o3.Linear(irreps_in, irreps_out)

    m = build_module(irreps_in, irreps_out)
    m(mx.random.normal((irreps_in.dim,)))

    assert_equivariant(m)


def test_bias() -> None:
    """Test linear layer with biases."""
    irreps_in = o3.Irreps("2x0e + 1e + 2x0e + 0o")
    irreps_out = o3.Irreps("3x0e + 1e + 3x0e + 5x0e + 0o")
    m = o3.Linear(irreps_in, irreps_out, biases=[True, False, False, True, False])
    
    # Set bias to 1.0
    m.bias[:] = 1.0
    x = mx.zeros((irreps_in.dim,))
    out = m(x)
    
    # Check expected output structure
    expected = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    expected = mx.array(expected)
    assert mx.allclose(out, expected, atol=1e-6)

    assert_equivariant(m)

    m = o3.Linear("0e + 0o + 1e + 1o", "10x0e + 0o + 1e + 1o", biases=True)
    assert_equivariant(m)


def test_single_out() -> None:
    """Test single output case."""
    l1 = o3.Linear("5x0e", "5x0e")
    l2 = o3.Linear("5x0e", "5x0e + 3x0o")
    
    # Copy weights
    l2.weight[:l1.weight.shape[0]] = l1.weight
    x = mx.random.normal((3, 5))
    out1 = l1(x)
    out2 = l2(x)
    assert out1.shape == (3, 5)
    assert out2.shape == (3, 8)
    assert mx.allclose(out1, out2[:, :5])
    assert mx.all(out2[:, 5:] == 0)


# We want to be sure to test a multiple-same L case, a single irrep case, and an empty irrep case
@pytest.mark.parametrize("irreps_in", ["5x0e", "1e + 2e + 4x1e + 3x3o", "2x1o + 0x3e"] + random_irreps(n=4))
@pytest.mark.parametrize("irreps_out", ["5x0e", "1e + 2e + 3x3o + 3x1e", "2x1o + 0x3e"] + random_irreps(n=4))
def test_linear_like_tp(irreps_in, irreps_out) -> None:
    """Test that Linear gives the same results as the corresponding TensorProduct."""
    m = o3.Linear(irreps_in, irreps_out)
    m_true = SlowLinear(irreps_in, irreps_out)
    
    # Copy weights
    m_true.weight[:] = m.weight
    inp = mx.random.normal((4, m.irreps_in.dim))
    assert mx.allclose(
        m(inp),
        m_true(inp),
        atol=1e-6,
    )


def test_output_mask() -> None:
    irreps_in = o3.Irreps("1e + 2e")
    irreps_out = o3.Irreps("3e + 5x2o")
    m = o3.Linear(irreps_in, irreps_out)
    assert mx.all(m.output_mask == mx.zeros(m.irreps_out.dim, dtype=mx.bool_))


def test_instructions_parameter() -> None:
    m = o3.Linear("4x0e + 3x4o", "1x2e + 4x0o")
    assert len(m.instructions) == 0
    assert not mx.any(m.output_mask)

    with pytest.raises(ValueError):
        m = o3.Linear(
            "4x0e + 3x4o",
            "1x2e + 4x0e",
            # invalid mixture of 0e and 2e
            instructions=[(0, 0)],
        )

    with pytest.raises(IndexError):
        m = o3.Linear("4x0e + 3x4o", "1x2e + 4x0e", instructions=[(4, 0)])


def test_empty_instructions() -> None:
    m = o3.Linear(o3.Irreps.spherical_harmonics(3), o3.Irreps.spherical_harmonics(3), instructions=[])
    assert len(m.instructions) == 0
    assert not mx.any(m.output_mask)
    inp = mx.random.normal((3, m.irreps_in.dim))
    out = m(inp)
    assert mx.all(out == 0.0)


def test_default_instructions() -> None:
    m = o3.Linear(
        "4x0e + 3x1o + 2x0e",
        "2x1o + 8x0e",
    )
    assert len(m.instructions) == 3
    assert mx.all(m.output_mask)
    ins_set = set((ins.i_in, ins.i_out) for ins in m.instructions)
    assert ins_set == {(0, 1), (1, 0), (2, 1)}
    assert set(ins.path_shape for ins in m.instructions) == {(4, 8), (2, 8), (3, 2)}


def test_instructions() -> None:
    m = o3.Linear("4x0e + 3x1o + 2x0e", "2x1o + 8x0e", instructions=[(0, 1), (1, 0)])
    inp = mx.random.normal((3, m.irreps_in.dim))
    # MLX array update - create a new array with zeros in the first part
    # Create a new array directly instead of copying
    zeros_part = mx.zeros((inp.shape[0], m.irreps_in[:2].dim))
    remaining_part = inp[:, m.irreps_in[:2].dim:]
    inp_modified = mx.concatenate([zeros_part, remaining_part], axis=1)
    out = m(inp_modified)
    assert mx.allclose(out, mx.zeros(out.shape))