# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import symforce.internal.symbolic as sf
from symforce import ops
from symforce import typing as T
from symforce.ops.interfaces.lie_group import LieGroup

from .matrix import Matrix
from .matrix import Matrix22
from .matrix import Vector2
from .rot2 import Rot2


class Pose2(LieGroup):
    """
    Group of two-dimensional rigid body transformations - R2 x SO(2).

    The storage space is a complex (real, imag) for rotation followed by a position (x, y).

    The tangent space is one angle for rotation followed by two elements for translation in the
    non-rotated frame.

    For Lie group enthusiasts: This class is on the PRODUCT manifold, if you really really want
    SE(2) you should use Pose2_SE2.  On this class, the group operations (e.g. compose and between)
    operate as you'd expect for a Pose or SE(2), but the manifold operations (e.g. retract and
    local_coordinates) operate on the product manifold SO(2) x R2.  This means that:

      - retract(a, vec) != compose(a, from_tangent(vec))

      - local_coordinates(a, b) != to_tangent(between(a, b))

      - There is no hat operator, because from_tangent/to_tangent is not the matrix exp/log
    """

    Pose2T = T.TypeVar("Pose2T", bound="Pose2")

    def __init__(self, R: Rot2 = None, t: Vector2 = None) -> None:
        """
        Construct from elements in SO2 and R2.
        """
        self.R = R or Rot2()
        self.t = t or Vector2()

        assert isinstance(self.R, Rot2)
        assert isinstance(self.t, Vector2)
        assert self.t.shape == (2, 1), self.t.shape

    def rotation(self) -> Rot2:
        """
        Accessor for the rotation component

        Does not make a copy.  Also accessible as `self.R`
        """
        return self.R

    def position(self) -> Vector2:
        """
        Accessor for the position component

        Does not make a copy.  Also accessible as `self.t`
        """
        return self.t

    # -------------------------------------------------------------------------
    # Storage concept - see symforce.ops.storage_ops
    # -------------------------------------------------------------------------

    def __repr__(self):
        # type: () -> str
        return "<{} R={}, t=({}, {})>".format(
            self.__class__.__name__, repr(self.R), repr(self.t[0]), repr(self.t[1])
        )

    @classmethod
    def storage_dim(cls) -> int:
        return Rot2.storage_dim() + Vector2.storage_dim()

    def to_storage(self) -> T.List[T.Scalar]:
        return self.R.to_storage() + self.t.to_storage()

    @classmethod
    def from_storage(cls, vec: T.Sequence[T.Scalar]) -> Pose2:
        assert len(vec) == cls.storage_dim()
        return cls(
            R=Rot2.from_storage(vec[0 : Rot2.storage_dim()]),
            t=Vector2(*vec[Rot2.storage_dim() : Rot2.storage_dim() + 2]),
        )

    # -------------------------------------------------------------------------
    # Group concept - see symforce.ops.group_ops
    # -------------------------------------------------------------------------

    @classmethod
    def identity(cls) -> Pose2:
        return cls(R=Rot2.identity(), t=Vector2.zero())

    def compose(self, other: Pose2) -> Pose2:
        assert isinstance(other, self.__class__)
        return self.__class__(R=self.R * other.R, t=self.t + self.R * other.t)

    def inverse(self) -> Pose2:
        R_inv = self.R.inverse()
        return self.__class__(R=R_inv, t=-(R_inv * self.t))

    # -------------------------------------------------------------------------
    # Lie group implementation
    # -------------------------------------------------------------------------

    @classmethod
    def tangent_dim(cls) -> int:
        return 3

    @classmethod
    def from_tangent(cls, v: T.Sequence[T.Scalar], epsilon: T.Scalar = sf.epsilon()) -> Pose2:
        theta = v[0]
        R = Rot2.from_tangent([theta], epsilon=epsilon)
        t = Vector2(v[1], v[2])
        return cls(R, t)

    def to_tangent(self, epsilon: T.Scalar = sf.epsilon()) -> T.List[T.Scalar]:
        # This uses atan2, so the resulting theta is between -pi and pi
        theta = self.R.to_tangent(epsilon=epsilon)[0]
        return [theta, self.t[0], self.t[1]]

    def storage_D_tangent(self) -> Matrix:
        """
        Note: generated from symforce/notebooks/storage_D_tangent.ipynb
        """
        storage_D_tangent_R = self.R.storage_D_tangent()
        storage_D_tangent_t = Matrix22.eye()
        return Matrix.block_matrix(
            [[storage_D_tangent_R, Matrix.zeros(2, 2)], [Matrix.zeros(2, 1), storage_D_tangent_t]]
        )

    def tangent_D_storage(self) -> Matrix:
        """
        Note: generated from symforce/notebooks/tangent_D_storage.ipynb
        """
        tangent_D_storage_R = self.R.tangent_D_storage()
        tangent_D_storage_t = Matrix22.eye()
        return Matrix.block_matrix(
            [[tangent_D_storage_R, Matrix.zeros(1, 2)], [Matrix.zeros(2, 2), tangent_D_storage_t]]
        )

    # NOTE(hayk, aaron): Override retract + local_coordinates, because we're treating
    # the Lie group as the product manifold of SO3 x R3 while leaving compose as normal
    # Pose3 composition.

    def retract(self: Pose2, vec: T.Sequence[T.Scalar], epsilon: T.Scalar = sf.epsilon()) -> Pose2:
        return Pose2(
            R=self.R.retract(vec[:1], epsilon=epsilon),
            t=ops.LieGroupOps.retract(self.t, vec[1:], epsilon=epsilon),
        )

    def local_coordinates(
        self: Pose2T, b: Pose2T, epsilon: T.Scalar = sf.epsilon()
    ) -> T.List[T.Scalar]:
        return self.R.local_coordinates(b.R, epsilon=epsilon) + ops.LieGroupOps.local_coordinates(
            self.t, b.t, epsilon=epsilon
        )

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    @T.overload
    def __mul__(self, right: Pose2) -> Pose2:  # pragma: no cover
        pass

    @T.overload
    def __mul__(self, right: Vector2) -> Vector2:  # pragma: no cover
        pass

    def __mul__(self, right: T.Union[Pose2, Vector2]) -> T.Union[Pose2, Vector2]:
        """
        Left-multiply with a compatible quantity.

        Args:
            right: (Pose2 | R2)

        Returns:
            (Pose2 | R2)
        """
        if isinstance(right, Vector2):
            assert right.shape == (2, 1), right.shape
            return self.R * right + self.t
        elif isinstance(right, self.__class__):
            return self.compose(right)
        else:
            raise NotImplementedError(f'Unsupported type: "{right}"')

    def to_homogenous_matrix(self) -> Matrix:
        """
        A matrix representation of this element in the Euclidean space that contains it.

        Returns:
            3x3 Matrix
        """
        R = self.R.to_rotation_matrix()
        return (R.row_join(self.t)).col_join(Matrix(1, 3, [0, 0, 1]))
