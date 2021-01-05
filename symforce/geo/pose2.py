from __future__ import annotations

from symforce.ops.interfaces.lie_group import LieGroup
from symforce import sympy as sm
from symforce import types as T

from .matrix import Matrix
from .matrix import Matrix13
from .matrix import Matrix21
from .matrix import Matrix33
from .matrix import Vector1
from .matrix import Vector2
from .matrix import Vector3
from .rot2 import Rot2


class Pose2(LieGroup):
    """
    Group of two-dimensional rigid body transformations - SE(2).

    The storage space is a complex (real, imag) for rotation followed by a position (x, y).

    The tangent space is two elements for translation followed by one angle for rotation.
    TODO(hayk): Flip this to match Pose3 with rotation first.
    """

    def __init__(self, R: Rot2 = None, t: Matrix = None) -> None:
        """
        Construct from elements in SO2 and R2.

        Args:
            R (Rot2):
            t (Matrix): 2x1 translation vector
        """
        self.R = R or Rot2()
        self.t = t or Vector2()

        assert isinstance(self.R, Rot2)
        assert isinstance(self.t, Vector2)
        assert self.t.shape == (2, 1), self.t.shape

    # -------------------------------------------------------------------------
    # Storage concept - see symforce.ops.storage_ops
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return "<Pose2 R={}, t=({}, {})>".format(repr(self.R), repr(self.t[0]), repr(self.t[1]))

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
    def from_tangent(cls, v: T.Sequence[T.Scalar], epsilon: T.Scalar = 0) -> Pose2:
        theta = v[2]
        R = Rot2.from_tangent([theta], epsilon=epsilon)

        a = (R.z.imag + epsilon * sm.sign_no_zero(R.z.imag)) / (
            theta + epsilon * sm.sign_no_zero(theta)
        )
        b = (1 - R.z.real) / (theta + epsilon * sm.sign_no_zero(theta))

        t = Vector2(a * v[0] - b * v[1], b * v[0] + a * v[1])
        return Pose2(R, t)

    def to_tangent(self, epsilon: T.Scalar = 0) -> T.List[T.Scalar]:

        # This uses atan2, so the resulting theta is between -pi and pi
        theta = self.R.to_tangent(epsilon=epsilon)[0]

        halftheta = 0.5 * (theta + sm.sign_no_zero(theta) * epsilon)
        a = (
            halftheta
            * (1 + self.R.z.real)
            / (self.R.z.imag + sm.sign_no_zero(self.R.z.imag) * epsilon)
        )

        V_inv = Matrix([[a, halftheta], [-halftheta, a]])
        t_tangent = V_inv * self.t
        return [t_tangent[0], t_tangent[1], theta]

    @classmethod
    def hat(cls, vec: T.List[T.Scalar]) -> Matrix33:
        t_tangent = [vec[0], vec[1]]
        R_tangent = [vec[2]]
        top_left = Rot2.hat(R_tangent)
        top_right = Matrix21(t_tangent)
        bottom = Matrix13.zero()
        return T.cast(Matrix33, top_left.row_join(top_right).col_join(bottom))

    def storage_D_tangent(self) -> Matrix:
        """
        Note: generated from symforce/notebooks/storage_D_tangent.ipynb
        """
        storage_D_tangent_R = self.R.storage_D_tangent()
        storage_D_tangent_t = self.R.to_rotation_matrix()
        return Matrix.block_matrix(
            [[Matrix.zeros(2, 2), storage_D_tangent_R], [storage_D_tangent_t, Matrix.zeros(2, 1)]]
        )

    def tangent_D_storage(self) -> Matrix:
        """
        Note: generated from symforce/notebooks/tangent_D_storage.ipynb
        """
        tangent_D_storage_R = self.R.tangent_D_storage()
        tangent_D_storage_t = self.R.to_rotation_matrix().T
        return Matrix.block_matrix(
            [[Matrix.zeros(2, 2), tangent_D_storage_t], [tangent_D_storage_R, Matrix.zeros(1, 2)]]
        )

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def __mul__(self, right: T.Union[Pose2, Matrix]) -> T.Union[Pose2, Matrix]:
        """
        Left-multiply with a compatible quantity.

        Args:
            right: (Pose2 | R2)

        Returns:
            (Pose2 | R2)
        """
        if isinstance(right, Matrix):
            assert right.shape == (2, 1), right.shape
            return self.R * right + self.t
        elif isinstance(right, Pose2):
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
