# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import symforce.internal.symbolic as sf
from symforce import typing as T
from symforce.ops.interfaces import LieGroup

from .matrix import Matrix
from .matrix import Matrix24
from .matrix import Matrix42
from .matrix import Vector3
from .rot3 import Rot3


class Unit3(LieGroup):
    """
    Direction in R^3, represented as a Rot3 that transforms [0, 0, 1] to the desired direction.
    The storage is therefore a quaternion and the tangent space is 2 dimensional.
    Most operations are implemented using operations from Rot3.

    Note: an alternative implementation could directly store a unit vector and define its boxplus
    manifold as described in Appendix B.2 of [Hertzberg 2013]. This can be done by finding the
    Householder reflector of x and use it to transform the exponential map of delta, which is a
    small perturbation in the tangent space (R^2). Namely,

    x.retract(delta) = x [+] delta = Rx * Exp(delta), where
    Exp(delta) = [sinc(||delta||) * delta, cos(||delta||)], and
    Rx = (I - 2 vv^T / (v^Tv))X, v = x - e_z != 0, X is a matrix negating 2nd vector component
       = I                     , x = e_z

    [Hertzberg 2013] Integrating Generic Sensor Fusion Algorithms with Sound State Representations
    through Encapsulation of Manifolds
    """

    E_Z = Vector3.unit_z()

    def __init__(self, rot3: Rot3 = None) -> None:
        """
        Construct from a rot3, or identity if none provided.
        """
        self.rot3 = rot3 if rot3 is not None else Rot3.identity()
        assert isinstance(self.rot3, Rot3)

    # -------------------------------------------------------------------------
    # Storage concept - see symforce.ops.storage_ops
    # -------------------------------------------------------------------------
    def __repr__(self) -> str:
        xyz = self.to_unit_vector()
        return "<Unit3 xyz=[{}, {}, {}]>".format(repr(xyz[0]), repr(xyz[1]), repr(xyz[2]))

    @classmethod
    def storage_dim(cls) -> int:
        return Rot3.storage_dim()

    def to_storage(self) -> T.List[T.Scalar]:
        return self.rot3.to_storage()

    @classmethod
    def from_storage(cls, vec: T.Sequence[T.Scalar]) -> Unit3:
        return cls(Rot3.from_storage(vec))

    @classmethod
    def symbolic(cls, name: str, **kwargs: T.Any) -> Unit3:
        return cls(Rot3.symbolic(name, **kwargs))

    # -------------------------------------------------------------------------
    # Group concept - see symforce.ops.group_ops
    # -------------------------------------------------------------------------

    @classmethod
    def identity(cls) -> Unit3:
        return cls(Rot3.identity())

    def compose(self, other: Unit3) -> Unit3:
        return Unit3(self.rot3.compose(other.rot3))

    def inverse(self) -> Unit3:
        return Unit3(self.rot3.inverse())

    # -------------------------------------------------------------------------
    # Lie group implementation
    # -------------------------------------------------------------------------

    @classmethod
    def tangent_dim(cls) -> int:
        return 2

    @classmethod
    def from_tangent(cls, v: T.Sequence[T.Scalar], epsilon: T.Scalar = sf.epsilon()) -> Unit3:
        return cls(Rot3.from_tangent([-v[1], v[0], sf.S.Zero], epsilon=epsilon))

    def to_tangent(self, epsilon: T.Scalar = sf.epsilon()) -> T.List[T.Scalar]:
        v = self.rot3.to_tangent(epsilon=epsilon)
        return [v[1], -v[0]]

    def storage_D_tangent(self) -> Matrix42:
        D = self.rot3.storage_D_tangent()
        return T.cast(Matrix42, Matrix.column_stack(D[:, 1], -D[:, 0]))

    def tangent_D_storage(self) -> Matrix24:
        return 4 * T.cast(Matrix24, self.storage_D_tangent().T)

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def to_rotation(self) -> Rot3:
        return self.rot3

    def to_unit_vector(self) -> Vector3:
        return self.rot3 * self.E_Z

    @classmethod
    def from_vector(cls, a: Vector3, epsilon: T.Scalar = sf.epsilon()) -> Unit3:
        """
        Return a Unit3 that points along the direction of vector a. a does not have to be a unit
        vector.
        """
        u = a.normalized(epsilon=epsilon)
        return cls(Rot3.from_two_unit_vectors(cls.E_Z, u, epsilon=epsilon))

    @classmethod
    def random(cls, epsilon: T.Scalar = sf.epsilon()) -> Unit3:
        """
        Generate a random element of Unit3, by generating a random rotation first and then rotating
        e_z to get a random direction.
        """
        return cls.from_vector(Rot3.random() * cls.E_Z, epsilon=epsilon)
