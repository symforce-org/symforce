# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

from symforce import typing as T
from symforce.ops.interfaces import Group

from .quaternion import Quaternion


class DualQuaternion(Group):
    """
    Dual quaternions can be used for rigid motions in 3D. Similar to the way that rotations in
    3D space can be represented by quaternions of unit length, rigid motions in 3D space can be
    represented by dual quaternions of unit length. This fact is used in theoretical kinematics,
    and in applications to 3D computer graphics, robotics and computer vision.

    References:

        https://en.wikipedia.org/wiki/Dual_quaternion
    """

    def __init__(self, real_q: Quaternion, inf_q: Quaternion) -> None:
        """
        Construct from two quaternions - a real one and an infinitesimal one.

        Args:
            real_q (Quaternion):
            inf_q (Quaternion):
        """
        self.real_q = real_q
        self.inf_q = inf_q

    # -------------------------------------------------------------------------
    # Storage concept - see symforce.ops.storage_ops
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return "<DQ real={}, inf={}>".format(repr(self.real_q), repr(self.inf_q))

    @classmethod
    def storage_dim(cls) -> int:
        return 2 * Quaternion.storage_dim()

    def to_storage(self) -> T.List[T.Scalar]:
        return self.real_q.to_storage() + self.inf_q.to_storage()

    @classmethod
    def from_storage(cls, vec: T.Sequence[T.Scalar]) -> DualQuaternion:
        assert len(vec) == cls.storage_dim()
        return cls(
            real_q=Quaternion.from_storage(vec[0 : Quaternion.storage_dim()]),
            inf_q=Quaternion.from_storage(vec[Quaternion.storage_dim() :]),
        )

    # -------------------------------------------------------------------------
    # Group concept - see symforce.ops.group_ops
    # -------------------------------------------------------------------------

    @classmethod
    def identity(cls) -> DualQuaternion:
        return cls(Quaternion.identity(), Quaternion.zero())

    def compose(self, other: DualQuaternion) -> DualQuaternion:
        return self.__class__(
            real_q=self.real_q * other.real_q,
            inf_q=self.real_q * other.inf_q + self.inf_q * other.real_q,
        )

    def inverse(self) -> DualQuaternion:
        return DualQuaternion(
            real_q=self.real_q.inverse(),
            inf_q=-self.real_q.inverse() * self.inf_q * self.real_q.inverse(),
        )

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def __mul__(self, right: DualQuaternion) -> DualQuaternion:
        """
        Left-multiply with another dual quaternion.

        Args:
            other (DualQuaternion):

        Returns:
            DualQuaternion:
        """
        return self.compose(right)

    def __div__(self, scalar: T.Scalar) -> DualQuaternion:
        """
        Scalar division.

        Args:
            scalar (Scalar):

        Returns:
            DualQuaternion:
        """
        return DualQuaternion(self.real_q / scalar, self.inf_q / scalar)

    __truediv__ = __div__

    def squared_norm(self) -> T.Scalar:
        """
        Squared norm when considering the dual quaternion as 8-tuple.

        Returns:
            Scalar:
        """
        return self.real_q.squared_norm() + self.inf_q.squared_norm()

    def conj(self) -> DualQuaternion:
        """
        Dual quaternion conjugate.

        Returns:
            DualQuaternion:
        """
        return DualQuaternion(self.real_q.conj(), self.inf_q.conj())
