"""
Mathematical utilities for e3nn operations.
"""

from ._reduce import germinate_formulas, reduce_permutation
from ._perm import (
    identity, compose, inverse, sign, to_cycles, germinate, group, is_group,
    rand, from_int, to_int, standard_representation, natural_representation
)
from ._orthonormalize import orthonormalize
from ._linalg import direct_sum, orthonormalize as orthonormalize_linalg, complete_basis
from ._soft_unit_step import soft_unit_step
from ._soft_one_hot_linspace import soft_one_hot_linspace
from ._normalize_activation import moment, Normalize2Mom, normalize2mom_class

__all__ = [
    "germinate_formulas",
    "reduce_permutation", 
    "identity",
    "compose", 
    "inverse",
    "sign",
    "to_cycles",
    "germinate",
    "group",
    "is_group",
    "rand",
    "from_int",
    "to_int",
    "standard_representation",
    "natural_representation",
    "orthonormalize",
    "direct_sum",
    "complete_basis",
    "soft_unit_step",
    "soft_one_hot_linspace",
    "moment",
    "Normalize2Mom",
    "normalize2mom_class",
]