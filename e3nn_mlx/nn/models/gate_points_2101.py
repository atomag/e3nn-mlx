"""Gate Points Network (v2101) — MLX Port

This is a lightweight port of e3nn.nn.models.gate_points_2101 to e3nn-mlx.
It implements a basic equivariant message-passing block with self-interactions,
gates, and a naive radius graph + scatter.
"""

import math
from typing import Dict, Optional, Tuple

import mlx.core as mx
from mlx import nn

from ... import o3
from ...math import soft_one_hot_linspace
from .._fc import FullyConnectedNet
from .._gate import Gate


def _scatter_add(src: mx.array, index: mx.array, dim_size: int) -> mx.array:
    """Scatter add along dim 0 using a boolean mask matmul trick.

    Args:
        src: (E, C)
        index: (E,)
        dim_size: N
    Returns:
        (N, C) with rows summed by index
    """
    E = src.shape[0]
    C = src.shape[1]
    if E == 0:
        return mx.zeros((dim_size, C), dtype=src.dtype)
    # Build mask (E, N): mask[e, j] = (index[e] == j)
    j = mx.arange(dim_size)[None, :]
    mask = (index[:, None] == j)
    mask = mask.astype(src.dtype)
    # out = mask^T @ src -> (N, E) @ (E, C) = (N, C)
    return mx.matmul(mask.transpose(1, 0), src)


def _radius_graph(pos: mx.array, r_max: float, batch: mx.array) -> Tuple[mx.array, mx.array]:
    """Naive radius graph with batching.

    Args:
        pos: (N, 3)
        r_max: float
        batch: (N,) graph ids
    Returns:
        (edge_src, edge_dst) each (E,)
    """
    N = pos.shape[0]
    if N == 0:
        return mx.zeros((0,), dtype=mx.int32), mx.zeros((0,), dtype=mx.int32)
    # Pairwise differences and distances
    diff = pos[:, None, :] - pos[None, :, :]
    d = mx.linalg.norm(diff, axis=-1)
    # Mask: within radius and not self
    mask = (d < r_max) & (d > 0)
    # Same-graph constraint
    same_graph = (batch[:, None] == batch[None, :])
    mask = mask & same_graph
    # Extract indices
    # Extract indices of True values via Python fallback
    import numpy as _np
    idx_pairs = []
    mask_list = mask.tolist()
    for i, row in enumerate(mask_list):
        for j, v in enumerate(row):
            if v:
                idx_pairs.append((i, j))
    if not idx_pairs:
        return mx.zeros((0,), dtype=mx.int32), mx.zeros((0,), dtype=mx.int32)
    nz = mx.array(idx_pairs, dtype=mx.int32)
    edge_src = nz[:, 0]
    edge_dst = nz[:, 1]
    return edge_src, edge_dst


class Convolution(nn.Module):
    """Equivariant convolution with radial MLP and tensor products (MLX).

    Parameters mirror the PyTorch version where applicable.
    """

    def __init__(
        self,
        irreps_in,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_out,
        number_of_basis: int,
        radial_layers: int,
        radial_neurons: int,
        num_neighbors: float,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_neighbors = num_neighbors

        # Self-connection and linear (node-attr) maps
        self.sc = o3.FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_out)
        self.lin1 = o3.FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_in)

        # Build mid irreps and TP instructions
        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_in):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()
        instructions = [(i1, i2, p[i_out], mode, train) for (i1, i2, i_out, mode, train) in instructions]

        tp = o3.TensorProduct(
            self.irreps_in,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )

        # Use linear radial MLP to avoid activation normalization coupling
        self.fc = FullyConnectedNet(
            [number_of_basis] + radial_layers * [radial_neurons] + [tp.weight_numel], act=None
        )
        self.tp = tp
        self.lin2 = o3.FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, self.irreps_out)

    def __call__(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedded) -> mx.array:
        weight = self.fc(edge_length_embedded)

        x = node_input
        s = self.sc(x, node_attr)
        x = self.lin1(x, node_attr)

        # Message passing: TP on edges, then scatter to destinations
        edge_features = self.tp(x[edge_src], edge_attr, weight)
        x = _scatter_add(edge_features, edge_dst, dim_size=x.shape[0]) / (self.num_neighbors ** 0.5)

        x = self.lin2(x, node_attr)

        # Blend self-connection and conv output with mask
        c_s, c_x = math.sin(math.pi / 8), math.cos(math.pi / 8)
        m = self.sc.output_mask
        c_x = (1 - m) + c_x * m
        return c_s * s + c_x * x


def _smooth_cutoff(x: mx.array) -> mx.array:
    u = 2 * (x - 1)
    y = (mx.cos(math.pi * u) * (-1.0) + 1.0) / 2.0
    y = mx.where(u > 0, mx.zeros_like(y), y)
    y = mx.where(u < -1, mx.ones_like(y), y)
    return y


def _tp_path_exists(irreps_in1, irreps_in2, ir_out) -> bool:
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)
    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


class _Compose(nn.Module):
    def __init__(self, first, second) -> None:
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def __call__(self, *args):
        x = self.first(*args)
        return self.second(x)


class Network(nn.Module):
    """Equivariant gate points network (v2101) — MLX.

    Arguments closely follow the PyTorch reference implementation.
    """

    def __init__(
        self,
        irreps_in: Optional[o3.Irreps],
        irreps_hidden: o3.Irreps,
        irreps_out: o3.Irreps,
        irreps_node_attr: Optional[o3.Irreps],
        irreps_edge_attr: Optional[o3.Irreps],
        layers: int,
        max_radius: float,
        number_of_basis: int,
        radial_layers: int,
        radial_neurons: int,
        num_neighbors: float,
        num_nodes: float,
        reduce_output: bool = True,
    ) -> None:
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.reduce_output = reduce_output

        self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr) if irreps_node_attr is not None else o3.Irreps("0e")
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)

        self.input_has_node_in = irreps_in is not None
        self.input_has_node_attr = irreps_node_attr is not None

        irreps = self.irreps_in if self.irreps_in is not None else o3.Irreps("0e")

        def _silu(v: mx.array) -> mx.array:
            return v * mx.sigmoid(v)
        act = {1: _silu, -1: mx.tanh}
        act_gates = {1: mx.sigmoid, -1: mx.tanh}

        self.layers = []
        for _ in range(layers):
            irreps_scalars = o3.Irreps([
                (mul, ir)
                for mul, ir in self.irreps_hidden
                if ir.l == 0 and _tp_path_exists(irreps, self.irreps_edge_attr, ir)
            ])
            irreps_gated = o3.Irreps([
                (mul, ir)
                for mul, ir in self.irreps_hidden
                if ir.l > 0 and _tp_path_exists(irreps, self.irreps_edge_attr, ir)
            ])
            ir = "0e" if _tp_path_exists(irreps, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars],
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],
                irreps_gated,
            )
            conv = Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors,
            )
            irreps = gate.irreps_out
            self.layers.append(_Compose(conv, gate))

        self.layers.append(
            Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                self.irreps_out,
                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors,
            )
        )

    def __call__(self, data: Dict[str, mx.array]) -> mx.array:
        # batch vector; default zeros if absent
        batch = data.get("batch", mx.zeros((data["pos"].shape[0],), dtype=mx.int32))

        # Build naive radius graph
        edge_src, edge_dst = _radius_graph(data["pos"], self.max_radius, batch)
        edge_vec = data["pos"][edge_src] - data["pos"][edge_dst]
        edge_sh = o3.spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization="component")
        edge_length = mx.linalg.norm(edge_vec, axis=1)
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length, start=0.0, end=self.max_radius, number=self.number_of_basis, basis="gaussian", cutoff=False
        ) * (self.number_of_basis ** 0.5)
        edge_attr = _smooth_cutoff(edge_length / self.max_radius)[:, None] * edge_sh

        # Prepare node input and attributes
        if self.input_has_node_in and "x" in data:
            x = data["x"]
        else:
            x = mx.ones((data["pos"].shape[0], 1), dtype=data["pos"].dtype)

        if self.input_has_node_attr and "z" in data:
            z = data["z"]
        else:
            z = mx.ones((data["pos"].shape[0], 1), dtype=data["pos"].dtype)

        # Apply layers
        for lay in self.layers:
            x = lay(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)

        # Reduce by graph if requested
        if self.reduce_output:
            out = _scatter_add(x, batch.astype(mx.int32), dim_size=int(mx.max(batch).item()) + 1)
            return out / (self.num_nodes ** 0.5)
        else:
            return x
