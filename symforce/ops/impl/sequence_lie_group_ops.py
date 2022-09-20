# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

from symforce import typing as T
from symforce.ops import LieGroupOps
from symforce.ops import StorageOps
from symforce.typing_util import get_type

from .sequence_group_ops import SequenceGroupOps

if T.TYPE_CHECKING:
    from symforce import geo


class SequenceLieGroupOps(SequenceGroupOps):
    @staticmethod
    def tangent_dim(a: T.SequenceElement) -> int:
        return sum(LieGroupOps.tangent_dim(v) for v in a)

    @staticmethod
    def from_tangent(
        a: T.SequenceElement, vec: T.Sequence[T.Scalar], epsilon: T.Scalar
    ) -> T.SequenceElement:
        assert len(vec) == SequenceLieGroupOps.tangent_dim(a)
        new_a = []
        inx = 0
        for v in a:
            dim = LieGroupOps.tangent_dim(v)
            new_a.append(LieGroupOps.from_tangent(v, vec[inx : inx + dim], epsilon))
            inx += dim
        return get_type(a)(new_a)

    @staticmethod
    def to_tangent(a: T.SequenceElement, epsilon: T.Scalar) -> T.List[T.Scalar]:
        return [x for v in a for x in LieGroupOps.to_tangent(v, epsilon)]

    @staticmethod
    def storage_D_tangent(a: T.SequenceElement) -> geo.Matrix:
        from symforce import geo

        mat = geo.Matrix(StorageOps.storage_dim(a), LieGroupOps.tangent_dim(a))
        s_inx = 0
        t_inx = 0
        for v in a:
            s_dim = StorageOps.storage_dim(v)
            t_dim = LieGroupOps.tangent_dim(v)
            mat[s_inx : s_inx + s_dim, t_inx : t_inx + t_dim] = LieGroupOps.storage_D_tangent(v)
            s_inx += s_dim
            t_inx += t_dim
        return mat

    @staticmethod
    def tangent_D_storage(a: T.SequenceElement) -> geo.Matrix:
        from symforce import geo

        mat = geo.Matrix(LieGroupOps.tangent_dim(a), StorageOps.storage_dim(a))
        t_inx = 0
        s_inx = 0
        for v in a:
            t_dim = LieGroupOps.tangent_dim(v)
            s_dim = StorageOps.storage_dim(v)
            mat[t_inx : t_inx + t_dim, s_inx : s_inx + s_dim] = LieGroupOps.tangent_D_storage(v)
            t_inx += t_dim
            s_inx += s_dim
        return mat

    @staticmethod
    def retract(
        a: T.SequenceElement, vec: T.Sequence[T.Scalar], epsilon: T.Scalar
    ) -> T.SequenceElement:
        assert len(vec) == SequenceLieGroupOps.tangent_dim(a)
        new_a = []
        inx = 0
        for v in a:
            dim = LieGroupOps.tangent_dim(v)
            new_a.append(LieGroupOps.retract(v, vec[inx : inx + dim], epsilon))
            inx += dim
        return get_type(a)(new_a)

    @staticmethod
    def local_coordinates(
        a: T.SequenceElement, b: T.SequenceElement, epsilon: T.Scalar
    ) -> T.List[T.Scalar]:
        return [x for va, vb in zip(a, b) for x in LieGroupOps.local_coordinates(va, vb, epsilon)]
