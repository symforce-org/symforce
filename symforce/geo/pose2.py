from symforce.ops.interfaces.lie_group import LieGroup
from symforce import sympy as sm
from symforce import types as T

from .matrix import Matrix
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

    def __init__(self, R=None, t=None):
        # type: (Rot2, Matrix) -> None
        """
        Construct from elements in SO2 and R2.

        Args:
            R (Rot2):
            t (Matrix): 2x1 translation vector
        """
        self.R = R or Rot2()
        self.t = t or Vector2()

        assert isinstance(self.R, Rot2)
        assert isinstance(self.t, sm.MatrixBase)
        assert self.t.shape == (2, 1), self.t.shape

    # -------------------------------------------------------------------------
    # Storage concept - see symforce.ops.storage_ops
    # -------------------------------------------------------------------------

    def __repr__(self):
        # type: () -> str
        return "<Pose2 R={}, t=({}, {})>".format(repr(self.R), repr(self.t[0]), repr(self.t[1]))

    @classmethod
    def storage_dim(cls):
        # type: () -> int
        return Rot2.storage_dim() + Vector2.storage_dim()

    def to_storage(self):
        # type: () -> T.List[T.Scalar]
        return self.R.to_storage() + self.t.to_storage()

    @classmethod
    def from_storage(cls, vec):
        # type: (T.Sequence[T.Scalar]) -> Pose2
        assert len(vec) == cls.storage_dim()
        return cls(
            R=Rot2.from_storage(vec[0 : Rot2.storage_dim()]),
            t=Vector2(*vec[Rot2.storage_dim() : Rot2.storage_dim() + 2]),
        )

    # -------------------------------------------------------------------------
    # Group concept - see symforce.ops.group_ops
    # -------------------------------------------------------------------------

    @classmethod
    def identity(cls):
        # type: () -> Pose2
        return cls(R=Rot2.identity(), t=Vector2.zero())

    def compose(self, other):
        # type: (Pose2) -> Pose2
        assert isinstance(other, self.__class__)
        return self.__class__(R=self.R * other.R, t=self.t + self.R * other.t)

    def inverse(self):
        # type: () -> Pose2
        R_inv = self.R.inverse()
        return self.__class__(R=R_inv, t=-(R_inv * self.t))

    # -------------------------------------------------------------------------
    # Lie group implementation
    # -------------------------------------------------------------------------

    @classmethod
    def tangent_dim(cls):
        # type: () -> int
        return 3

    @classmethod
    def from_tangent(cls, v, epsilon=0):
        # type: (T.Sequence[T.Scalar], T.Scalar) -> Pose2
        theta = v[2]
        R = Rot2.from_tangent([theta], epsilon=epsilon)

        a = R.z.imag / (theta + epsilon)
        b = (1 - R.z.real) / (theta + epsilon)

        t = Vector2(a * v[0] - b * v[1], b * v[0] + a * v[1])
        return Pose2(R, t)

    def to_tangent(self, epsilon=0):
        # type: (T.Scalar) -> T.List[T.Scalar]
        theta = self.R.to_tangent(epsilon=epsilon)[0]
        halftheta = 0.5 * theta
        a = (halftheta * self.R.z.imag) / sm.Max(epsilon, 1 - self.R.z.real)

        V_inv = Matrix([[a, halftheta], [-halftheta, a]])
        t_tangent = V_inv * self.t
        return [t_tangent[0], t_tangent[1], theta]

    @classmethod
    def hat(cls, vec):
        # type: (T.List[T.Scalar]) -> T.List[T.Scalar]
        t_tangent = Vector2(vec[0], vec[1])
        R_tangent = Vector1(vec[2])
        return Matrix(Rot2.hat(R_tangent)).row_join(t_tangent).col_join(Matrix.zeros(1, 3)).tolist()

    def storage_D_tangent(self):
        # type: () -> Matrix
        storage_D_tangent_R = self.R.storage_D_tangent()
        storage_D_tangent_t = self.R.to_rotation_matrix()
        return Matrix.block_matrix(
            [[Matrix.zeros(2, 2), storage_D_tangent_R], [storage_D_tangent_t, Matrix.zeros(2, 1)]]
        )

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def __mul__(self, right):
        # type: (T.Union[Pose2, Matrix]) -> T.Union[Pose2, Matrix]
        """
        Left-multiply with a compatible quantity.

        Args:
            right: (Pose2 | R2)

        Returns:
            (Pose2 | R2)
        """
        if isinstance(right, sm.MatrixBase):
            assert right.shape == (2, 1), right.shape
            return self.R * right + self.t
        elif isinstance(right, Pose2):
            return self.compose(right)
        else:
            raise NotImplementedError('Unsupported type: "{}"'.format(right))

    def to_homogenous_matrix(self):
        # type: () -> Matrix
        """
        A matrix representation of this element in the Euclidean space that contains it.

        Returns:
            3x3 Matrix
        """
        R = self.R.to_rotation_matrix()
        return (R.row_join(self.t)).col_join(Matrix(1, 3, [0, 0, 1]))
