from .base import Group
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

    STORAGE_DIM = 2 * Quaternion.STORAGE_DIM

    def __init__(self, real_q, inf_q):
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

    def __repr__(self):
        return "<DQ real={}, inf={}>".format(repr(self.real_q), repr(self.inf_q))

    def to_storage(self):
        return self.real_q.to_storage() + self.inf_q.to_storage()

    @classmethod
    def from_storage(cls, vec):
        assert len(vec) == cls.STORAGE_DIM
        return cls(
            real_q=Quaternion.from_storage(vec[0 : Quaternion.STORAGE_DIM]),
            inf_q=Quaternion.from_storage(vec[Quaternion.STORAGE_DIM :]),
        )

    # -------------------------------------------------------------------------
    # Group concept - see symforce.ops.group_ops
    # -------------------------------------------------------------------------

    @classmethod
    def identity(cls):
        return cls(Quaternion.identity(), Quaternion.zero())

    def compose(self, other):
        return self.__class__(
            real_q=self.real_q * other.real_q,
            inf_q=self.real_q * other.inf_q + self.inf_q * other.real_q,
        )

    def inverse(self):
        return DualQuaternion(
            real_q=self.real_q.inverse(),
            inf_q=-self.real_q.inverse() * self.inf_q * self.real_q.inverse(),
        )

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def __mul__(self, right):
        """
        Left-multiply with another dual quaternion.

        Args:
            other (DualQuaternion):

        Returns:
            DualQuaternion:
        """
        return self.compose(right)

    def __truediv__(self, scalar):
        """
        Scalar division.

        Args:
            scalar (Scalar):

        Returns:
            DualQuaternion:
        """
        return DualQuaternion(self.real_q / scalar, self.inf_q / scalar)

    def squared_norm(self):
        """
        Squared norm when considering the dual quaternion as 8-tuple.

        Returns:
            Scalar:
        """
        return self.real_q.squared_norm() + self.inf_q.squared_norm()

    def conj(self):
        """
        Dual quaternion conjugate.

        Returns:
            DualQuaternion:
        """
        return DualQuaternion(self.real_q.conj(), self.inf_q.conj())
