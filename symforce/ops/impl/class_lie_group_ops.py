# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import symforce.internal.symbolic as sf
from symforce import typing as T

from .class_group_ops import ClassGroupOps

if T.TYPE_CHECKING:
    from symforce import geo


class ClassLieGroupOps(ClassGroupOps):
    @staticmethod
    def tangent_dim(a: T.ElementOrType) -> int:
        return a.tangent_dim()

    @staticmethod
    def from_tangent(a: T.ElementOrType, vec: T.List[T.Scalar], epsilon: T.Scalar) -> T.Element:
        return a.from_tangent(vec, epsilon)

    @staticmethod
    def to_tangent(a: T.Element, epsilon: T.Scalar) -> T.List[T.Scalar]:
        return a.to_tangent(epsilon)

    @staticmethod
    def storage_D_tangent(a: T.Element) -> geo.Matrix:
        return a.storage_D_tangent()

    @staticmethod
    def tangent_D_storage(a: T.Element) -> geo.Matrix:
        return a.tangent_D_storage()

    @staticmethod
    def retract(a: T.Element, vec: T.Sequence[T.Scalar], epsilon: T.Scalar) -> T.Element:
        return a.retract(vec, epsilon)

    @staticmethod
    def local_coordinates(
        a: T.Element, b: T.Element, epsilon: T.Scalar = sf.epsilon()
    ) -> T.List[T.Scalar]:
        return a.local_coordinates(b, epsilon)
