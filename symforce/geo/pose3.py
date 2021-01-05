from __future__ import annotations

from symforce.ops.interfaces import LieGroup
from symforce import sympy as sm
from symforce import types as T

from .matrix import Matrix
from .matrix import Matrix14
from .matrix import Matrix31
from .matrix import Matrix44
from .matrix import Vector3
from .rot3 import Rot3


class Pose3(LieGroup):
    """
    Group of three-dimensional rigid body transformations - SE(3).

    The storage is a quaternion (x, y, z, w) for rotation followed by position (x, y, z).

    The tangent space is 3 elements for rotation followed by 3 elements for translation in
    the rotated frame, meaning we interpolate the translation in the tangent of the rotating
    frame for lie operations. This can be useful but is more expensive than SO3 x R3 for often
    no benefit.
    """

    def __init__(self, R: Rot3 = None, t: Matrix31 = None) -> None:
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

    # -------------------------------------------------------------------------
    # Storage concept - see symforce.ops.storage_ops
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return "<Pose3 R={}, t=({}, {}, {})>".format(
            repr(self.R), repr(self.t[0]), repr(self.t[1]), repr(self.t[2])
        )

    @classmethod
    def storage_dim(cls) -> int:
        return Rot3.storage_dim() + Vector3.storage_dim()

    def to_storage(self) -> T.List[T.Scalar]:
        return self.R.to_storage() + self.t.to_storage()

    @classmethod
    def from_storage(cls, vec: T.Sequence[T.Scalar]) -> Pose3:
        assert len(vec) == cls.storage_dim()
        return cls(
            R=Rot3.from_storage(vec[0 : Rot3.storage_dim()]),
            t=Vector3(*vec[Rot3.storage_dim() : Rot3.storage_dim() + 3]),
        )

    # -------------------------------------------------------------------------
    # Group concept - see symforce.ops.group_ops
    # -------------------------------------------------------------------------

    @classmethod
    def identity(cls) -> Pose3:
        return cls(R=Rot3.identity(), t=Vector3.zero())

    def compose(self, other: Pose3) -> Pose3:
        assert isinstance(other, self.__class__)
        return self.__class__(R=self.R * other.R, t=self.t + self.R * other.t)

    def inverse(self) -> Pose3:
        so3_inv = self.R.inverse()
        return self.__class__(R=so3_inv, t=-(so3_inv * self.t))

    # -------------------------------------------------------------------------
    # Lie group implementation
    # -------------------------------------------------------------------------

    @classmethod
    def tangent_dim(cls) -> int:
        return 6

    @classmethod
    def from_tangent(cls, v: T.Sequence[T.Scalar], epsilon: T.Scalar = 0) -> Pose3:
        R_tangent = (v[0], v[1], v[2])
        t_tangent_vector = Vector3(v[3], v[4], v[5])

        R = Rot3.from_tangent(R_tangent, epsilon=epsilon)
        R_hat = Rot3.hat(R_tangent)
        R_hat_sq = R_hat * R_hat
        R_tangent_vector = Vector3(R_tangent)
        theta = sm.sqrt(R_tangent_vector.squared_norm() + epsilon ** 2)

        V = (
            Matrix.eye(3)
            + (1 - sm.cos(theta)) / (theta ** 2) * R_hat
            + (theta - sm.sin(theta)) / (theta ** 3) * R_hat_sq
        )

        return cls(R, V * t_tangent_vector)

    def to_tangent(self, epsilon: T.Scalar = 0) -> T.List[T.Scalar]:
        R_tangent = self.R.to_tangent(epsilon=epsilon)
        R_tangent_vector = Vector3(R_tangent)
        theta = sm.sqrt(R_tangent_vector.squared_norm() + epsilon)
        R_hat = Rot3.hat(R_tangent)

        half_theta = 0.5 * theta

        V_inv = (
            Matrix.eye(3)
            - 0.5 * R_hat
            + (1 - theta * sm.cos(half_theta) / (2 * sm.sin(half_theta)))
            / (theta * theta)
            * (R_hat * R_hat)
        )
        t_tangent = V_inv * self.t
        return R_tangent_vector.col_join(t_tangent).to_flat_list()

    @classmethod
    def hat(cls, vec: T.List) -> Matrix44:
        R_tangent = [vec[0], vec[1], vec[2]]
        t_tangent = [vec[3], vec[4], vec[5]]
        top_left = Rot3.hat(R_tangent)
        top_right = Matrix31(t_tangent)
        bottom = Matrix14.zero()
        return T.cast(Matrix44, top_left.row_join(top_right).col_join(bottom))

    def storage_D_tangent(self) -> Matrix:
        """
        Note: generated from symforce/notebooks/storage_D_tangent.ipynb
        """
        storage_D_tangent_R = self.R.storage_D_tangent()
        storage_D_tangent_t = self.R.to_rotation_matrix()
        return Matrix.block_matrix(
            [[storage_D_tangent_R, Matrix.zeros(4, 3)], [Matrix.zeros(3, 3), storage_D_tangent_t],]
        )

    def tangent_D_storage(self) -> Matrix:
        """
        Note: generated from symforce/notebooks/tangent_D_storage.ipynb
        """
        tangent_D_storage_R = self.R.tangent_D_storage()
        tangent_D_storage_t = self.R.to_rotation_matrix().T
        return Matrix.block_matrix(
            [[tangent_D_storage_R, Matrix.zeros(3, 3)], [Matrix.zeros(3, 4), tangent_D_storage_t],]
        )

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def __mul__(self, right: T.Union[Pose3, Matrix]) -> T.Any:
        """
        Left-multiply with a compatible quantity.
        """
        if isinstance(right, Matrix):
            assert right.shape == (3, 1), right.shape
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
