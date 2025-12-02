# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

from symforce import typing as T
from symforce.ops.storage_ops import StorageOps as Ops

from .pair import Pair
from .pair import get_memtype

"""
This module contains functions for converting between different memory layouts.

Caspar uses an array of structs of arrays (AOSOA) memory layout.
This is because global memory access supports up to 16 bytes per thread.

The innermost structs are either float, float2, float3, or float4.

Example::

    poses_stacked = [x0, x1, x2, x3, x4, x5, x6,
                     y0, y1, y2, y3, y4, y5, y6,
                     z0, z1, z2, z3, z4, z5, z6]

    poses_caspar  = [x0, x1, x2, x3,
                     y0, y1, y2, y3,
                     z0, z1, z2, z3,
                     x4, x5, x6, __,
                     y4, y5, y6, __,
                     z4, z5, z6, __]
"""


def get_default_caspar_layout(stype: T.Union[T.StorableOrType, Pair]) -> list[list[int]]:
    """
    Default layout for a given storage type.
    """
    if isinstance(stype, Pair):
        return get_default_caspar_layout(get_memtype(stype))
    size = Ops.storage_dim(stype)
    layout = [list(range(i, min(i + 4, size))) for i in range(0, size, 4)]
    return layout


def caspar_size(size: int | T.StorableOrType) -> int:
    """
    Number of elements the caspar layout uses for a given size.
    """
    if not isinstance(size, int):
        size = Ops.storage_dim(size)
    return size if size % 4 != 3 else size + 1


def stacked_size(size: int | T.StorableOrType) -> int:
    """
    Number of elements the caspar layout uses for a given size.
    """
    if not isinstance(size, int):
        size = Ops.storage_dim(size)
    return size


def chunk_dim(chunk: list[int]) -> int:
    """
    The number of elements in a chunk of the caspar layout.

    In the cuda kernels we use the float, float2, float3, float4 types.
    float3 is aligned to 16 bytes and needs padding.
    """
    assert len(chunk) <= 4
    return caspar_size(len(chunk))
