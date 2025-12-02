# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

from symforce import typing as T
from symforce.ops import StorageOps
from symforce.ops.interfaces import LieGroup

from ..code_generation.factor import Factor
from ..memory import caspar_size


def fac_key(fac: Factor, suffix: str = "") -> str:
    return f"facs__{fac.name}__{suffix}_"


def arg_key(fac: Factor, arg: str, suffix: str = "") -> str:
    return f"{fac_key(fac, 'args')[:-1]}__{arg}__{suffix}_"


def solver_key(suffix: str = "") -> str:
    return f"solver__{suffix}_"


def node_key(typ: T.LieGroupElement, suffix: str = "") -> str:
    return f"nodes__{typ.__name__}__{suffix}_"


def name_key(thing: T.LieGroupElement | Factor) -> str:
    if isinstance(thing, LieGroup) or isinstance(thing, type):
        return thing.__name__  # type: ignore[union-attr]
    elif isinstance(thing, Factor):
        return thing.name
    else:
        raise ValueError(f"Unknown type {type(thing)}")


def num_key(thing: T.LieGroupElement | Factor, suffix: str = "num_") -> str:
    return f"{name_key(thing)}_{suffix}"


def num_max_key(thing: T.LieGroupElement | Factor) -> str:
    return num_key(thing, "num_max_")


def num_blocks_key(thing: T.LieGroupElement | Factor) -> str:
    return num_key(thing, "bnum_")


def num_arg_key(thing: T.LieGroupElement | Factor) -> str:
    return num_key(thing, "num_max")


def max_stacked_storage(*things: T.LieGroupElement) -> str:
    inner = " , ".join(f" {StorageOps.storage_dim(n)} * {num_max_key(n)}" for n in things)
    return f"std::max({{{inner}}})"


class MemDesc:
    """
    Memory descriptor used to make the memory layout of the generated solver.
    """

    def __init__(
        self,
        dim: int,
        num_key: str | int,
        is_caspar_data: bool = True,
        dtype: str = "float",
        alignment: int = 16,
    ):
        assert alignment in set((4, 16))
        self.dim = dim
        self.dim_real = caspar_size(dim) if is_caspar_data else dim
        self.num_key = num_key
        self.dtype = dtype
        self.alignment = alignment
