# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

from .abstract_vector_group_ops import AbstractVectorGroupOps

from symforce import typing as T

if T.TYPE_CHECKING:
    from symforce import geo


class AbstractVectorLieGroupOps(AbstractVectorGroupOps):
    """
    An abstract base class for LieGroupOps implementations whose group operation
    is equivalent to storage representation addition, and whose identity element
    is the element whose storage representation is the 0 vector.

    For a list of abstract methods which child classes must define, see
    AbstractStorageOps from abstract_storage_ops.py
    """

    @classmethod
    def tangent_dim(cls, a: T.ElementOrType) -> int:
        return cls.storage_dim(a)

    @classmethod
    def from_tangent(
        cls, a: T.ElementOrType, vec: T.Sequence[T.Scalar], epsilon: T.Scalar
    ) -> T.Element:
        return cls.from_storage(a, vec)

    @classmethod
    def to_tangent(cls, a: T.Element, epsilon: T.Scalar) -> T.List[T.Scalar]:
        return cls.to_storage(a)

    @classmethod
    def storage_D_tangent(cls, a: T.Element) -> geo.Matrix:
        from symforce import geo

        return geo.Matrix.eye(cls.storage_dim(a))

    @classmethod
    def tangent_D_storage(cls, a: T.Element) -> geo.Matrix:
        from symforce import geo

        return geo.Matrix.eye(cls.storage_dim(a))

    @classmethod
    def retract(cls, a: T.Element, vec: T.Sequence[T.Scalar], epsilon: T.Scalar) -> T.Element:
        return cls.compose(a, cls.from_tangent(a, vec, epsilon))

    @classmethod
    def local_coordinates(cls, a: T.Element, b: T.Element, epsilon: T.Scalar) -> T.List[T.Scalar]:
        return cls.to_tangent(cls.compose(cls.inverse(a), b), epsilon)
