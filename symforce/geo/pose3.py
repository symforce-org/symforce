# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import symforce.internal.symbolic as sf
from symforce import ops
from symforce import typing as T
from symforce.ops.interfaces import LieGroup

from .matrix import Matrix
from .matrix import Matrix33
from .matrix import Vector3
from .rot3 import Rot3


class Pose3(LieGroup):
    """
    Group of three-dimensional rigid body transformations - SO(3) x R3.

    The storage is a quaternion (x, y, z, w) for rotation followed by position (x, y, z).

    The tangent space is 3 elements for rotation followed by 3 elements for translation in the
    non-rotated frame.

    For Lie group enthusiasts: This class is on the PRODUCT manifold, if you really really want
    SE(3) you should use Pose3_SE3.  On this class, the group operations (e.g. compose and between)
    operate as you'd expect for a Pose or SE(3), but the manifold operations (e.g. retract and
    local_coordinates) operate on the product manifold SO(3) x R3.  This means that:

      - retract(a, vec) != compose(a, from_tangent(vec))

      - local_coordinates(a, b) != to_tangent(between(a, b))

      - There is no hat operator, because from_tangent/to_tangent is not the matrix exp/log
    """

    Pose3T = T.TypeVar("Pose3T", bound="Pose3")

    def __init__(self, R: Rot3 = None, t: Vector3 = None) -> None:
        """
        Construct from elements in SO3 and R3.

        Args:
            R: Frame orientation
            t: Translation 3-vector in the global frame
        """
        self.R = R or Rot3()
        self.t = t or Vector3()

        assert isinstance(self.R, Rot3)
        assert isinstance(self.t, Vector3)
        assert self.t.shape == (3, 1), self.t.shape

    def rotation(self) -> Rot3:
        """
        Accessor for the rotation component

        Does not make a copy.  Also accessible as `self.R`
        """
        return self.R

    def position(self) -> Vector3:
        """
        Accessor for the position component

        Does not make a copy.  Also accessible as `self.t`
        """
        return self.t

    # -------------------------------------------------------------------------
    # Storage concept - see symforce.ops.storage_ops
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return "<{} R={}, t=({}, {}, {})>".format(
            self.__class__.__name__, repr(self.R), repr(self.t[0]), repr(self.t[1]), repr(self.t[2])
        )

    @classmethod
    def storage_dim(cls) -> int:
        return Rot3.storage_dim() + Vector3.storage_dim()

    def to_storage(self) -> T.List[T.Scalar]:
        return self.R.to_storage() + self.t.to_storage()

    @classmethod
    def from_storage(cls: T.Type[Pose3T], vec: T.Sequence[T.Scalar]) -> Pose3T:
        assert len(vec) == cls.storage_dim()
        return cls(
            R=Rot3.from_storage(vec[0 : Rot3.storage_dim()]),
            t=Vector3(*vec[Rot3.storage_dim() : Rot3.storage_dim() + 3]),
        )

    @classmethod
    def symbolic(cls: T.Type[Pose3T], name: str, **kwargs: T.Any) -> Pose3T:
        return cls(R=Rot3.symbolic(f"{name}.R"), t=Vector3.symbolic(f"{name}.t"))

    # -------------------------------------------------------------------------
    # Group concept - see symforce.ops.group_ops
    # -------------------------------------------------------------------------

    @classmethod
    def identity(cls: T.Type[Pose3T]) -> Pose3T:
        return cls(R=Rot3.identity(), t=Vector3.zero())

    def compose(self: Pose3T, other: Pose3T) -> Pose3T:
        assert isinstance(other, self.__class__)
        return self.__class__(R=self.R * other.R, t=self.t + self.R * other.t)

    def inverse(self: Pose3T) -> Pose3T:
        so3_inv = self.R.inverse()
        return self.__class__(R=so3_inv, t=-(so3_inv * self.t))

    # -------------------------------------------------------------------------
    # Lie group implementation
    # -------------------------------------------------------------------------

    @classmethod
    def tangent_dim(cls) -> int:
        return 6

    @classmethod
    def from_tangent(
        cls: T.Type[Pose3T], v: T.Sequence[T.Scalar], epsilon: T.Scalar = sf.epsilon()
    ) -> Pose3T:
        R_tangent = (v[0], v[1], v[2])
        t_tangent_vector = Vector3(v[3], v[4], v[5])

        R = Rot3.from_tangent(R_tangent, epsilon=epsilon)
        return cls(R, t_tangent_vector)

    def to_tangent(self: Pose3T, epsilon: T.Scalar = sf.epsilon()) -> T.List[T.Scalar]:
        R_tangent = Vector3(self.R.to_tangent(epsilon=epsilon))
        return R_tangent.col_join(self.t).to_flat_list()

    def storage_D_tangent(self: Pose3T) -> Matrix:
        """
        Note: generated from symforce/notebooks/storage_D_tangent.ipynb
        """
        storage_D_tangent_R = self.R.storage_D_tangent()
        storage_D_tangent_t = Matrix33.eye()
        return Matrix.block_matrix(
            [[storage_D_tangent_R, Matrix.zeros(4, 3)], [Matrix.zeros(3, 3), storage_D_tangent_t]]
        )

    def tangent_D_storage(self: Pose3T) -> Matrix:
        """
        Note: generated from symforce/notebooks/tangent_D_storage.ipynb
        """
        tangent_D_storage_R = self.R.tangent_D_storage()
        tangent_D_storage_t = Matrix33.eye()
        return Matrix.block_matrix(
            [[tangent_D_storage_R, Matrix.zeros(3, 3)], [Matrix.zeros(3, 4), tangent_D_storage_t]]
        )

    # NOTE(hayk, aaron): Override retract + local_coordinates, because we're treating
    # the Lie group as the product manifold of SO3 x R3 while leaving compose as normal
    # Pose3 composition.

    def retract(
        self: Pose3T, vec: T.Sequence[T.Scalar], epsilon: T.Scalar = sf.epsilon()
    ) -> Pose3T:
        return self.__class__(
            R=self.R.retract(vec[:3], epsilon=epsilon),
            t=ops.LieGroupOps.retract(self.t, vec[3:], epsilon=epsilon),
        )

    def local_coordinates(
        self: Pose3T, b: Pose3T, epsilon: T.Scalar = sf.epsilon()
    ) -> T.List[T.Scalar]:
        return self.R.local_coordinates(b.R, epsilon=epsilon) + ops.LieGroupOps.local_coordinates(
            self.t, b.t, epsilon=epsilon
        )

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    @T.overload
    def __mul__(self: Pose3T, right: Pose3T) -> Pose3T:  # pragma: no cover
        pass

    @T.overload
    def __mul__(self: Pose3T, right: Vector3) -> Vector3:  # pragma: no cover
        pass

    def __mul__(self: Pose3T, right: T.Union[Pose3T, Vector3]) -> T.Union[Pose3T, Vector3]:
        """
        Left-multiply with a compatible quantity.
        """
        if isinstance(right, Vector3):
            return self.R * right + self.t
        elif isinstance(right, Pose3):
            return self.compose(right)
        else:
            raise NotImplementedError(f'Unsupported type: "{right}"')

    def to_homogenous_matrix(self) -> Matrix:
        """
        4x4 matrix representing this pose transform.
        """
        R = self.R.to_rotation_matrix()
        return (R.row_join(self.t)).col_join(Matrix(1, 4, [0, 0, 0, 1]))
