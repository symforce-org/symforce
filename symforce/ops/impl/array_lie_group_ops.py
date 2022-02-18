# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from symforce.ops import StorageOps
from symforce.ops import LieGroupOps
from symforce import typing as T

from .array_group_ops import ArrayGroupOps

if T.TYPE_CHECKING:
    from symforce import geo


class ArrayLieGroupOps(ArrayGroupOps):
    """
    Implements Lie Group operations for numpy ndarrays.
    """

    @staticmethod
    def tangent_dim(a: T.ArrayElement) -> int:
        return sum([LieGroupOps.tangent_dim(v) for v in a])

    @staticmethod
    def from_tangent(
        a: T.ArrayElement, vec: T.Sequence[T.Scalar], epsilon: T.Scalar
    ) -> T.ArrayElement:
        assert len(vec) == ArrayLieGroupOps.tangent_dim(a)
        new_a = []
        inx = 0
        for v in a:
            dim = LieGroupOps.tangent_dim(v)
            new_a.append(LieGroupOps.from_tangent(v, vec[inx : inx + dim], epsilon))
            inx += dim
        return np.array(new_a).reshape(a.shape)

    @staticmethod
    def to_tangent(a: T.ArrayElement, epsilon: T.Scalar) -> T.List[T.Scalar]:
        return [x for v in a for x in LieGroupOps.to_tangent(v, epsilon)]

    @staticmethod
    def storage_D_tangent(a: T.ArrayElement) -> geo.Matrix:
        from symforce import geo

        return geo.M.eye(StorageOps.storage_dim(a), LieGroupOps.tangent_dim(a))

    @staticmethod
    def tangent_D_storage(a: T.ArrayElement) -> geo.Matrix:
        from symforce import geo

        return geo.M.eye(StorageOps.storage_dim(a), LieGroupOps.tangent_dim(a))

    @staticmethod
    def retract(a: T.ArrayElement, vec: T.Sequence[T.Scalar], epsilon: T.Scalar) -> T.ArrayElement:
        assert len(vec) == ArrayLieGroupOps.tangent_dim(a)
        new_a = []
        inx = 0
        for v in a:
            dim = LieGroupOps.tangent_dim(v)
            new_a.append(LieGroupOps.retract(v, vec[inx : inx + dim], epsilon))
            inx += dim
        return np.array(new_a).reshape(a.shape)

    @staticmethod
    def local_coordinates(
        a: T.ArrayElement, b: T.ArrayElement, epsilon: T.Scalar = 0
    ) -> T.List[T.Scalar]:
        return [x for va, vb in zip(a, b) for x in LieGroupOps.local_coordinates(va, vb, epsilon)]
