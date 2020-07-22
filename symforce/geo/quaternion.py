from __future__ import division
import numpy as np

from symforce import ops
from symforce.ops.interfaces import Group
from symforce import sympy as sm
from symforce import types as T

from .matrix import Matrix
from .matrix import Vector3


class Quaternion(Group):
    """
    Unit quaternions, also known as versors, provide a convenient mathematical notation for
    representing orientations and rotations of objects in three dimensions. Compared to Euler
    angles they are simpler to compose and avoid the problem of gimbal lock. Compared to rotation
    matrices they are more compact, more numerically stable, and more efficient.

    Storage is (x, y, z, w).

    References:

        https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    """

    def __init__(self, xyz, w):
        # type: (Matrix, T.Scalar) -> None
        """
        Construct from a real scalar and an imaginary unit vector.

        Args:
            xyz (Matrix): 3x1 vector
            w (Scalar):
        """
        assert isinstance(xyz, sm.MatrixBase)
        assert xyz.shape == (3, 1), xyz.shape
        self.xyz = xyz
        self.w = w

    @property
    def x(self):
        # type: () -> T.Scalar
        return self.xyz[0]

    @property
    def y(self):
        # type: () -> T.Scalar
        return self.xyz[1]

    @property
    def z(self):
        # type: () -> T.Scalar
        return self.xyz[2]

    # -------------------------------------------------------------------------
    # Storage concept - see symforce.ops.storage_ops
    # -------------------------------------------------------------------------

    def __repr__(self):
        # type: () -> str
        return "<Q xyzw=[{}, {}, {}, {}]>".format(
            repr(self.x), repr(self.y), repr(self.z), repr(self.w)
        )

    @classmethod
    def storage_dim(cls):
        # type: () -> int
        return 4

    def to_storage(self):
        # type: () -> T.List[T.Scalar]
        return [self.x, self.y, self.z, self.w]

    @classmethod
    def from_storage(cls, vec):
        # type: (T.Sequence[T.Scalar]) -> Quaternion
        assert len(vec) == cls.storage_dim()
        return cls(xyz=Matrix(vec[0:3]), w=vec[3])

    @classmethod
    def symbolic(cls, name, **kwargs):
        # type: (str, T.Any) -> Quaternion
        return cls.from_storage(
            [sm.Symbol("{}_{}".format(name, v), **kwargs) for v in ["x", "y", "z", "w"]]
        )

    # -------------------------------------------------------------------------
    # Group concept - see symforce.ops.group_ops
    # -------------------------------------------------------------------------

    @classmethod
    def identity(cls):
        # type: () -> Quaternion
        return cls(xyz=Vector3(0, 0, 0), w=1)

    def compose(self, other):
        # type: (Quaternion) -> Quaternion
        return self.__class__(
            xyz=self.w * other.xyz + other.w * self.xyz + self.xyz.cross(other.xyz),
            w=self.w * other.w - self.xyz.dot(other.xyz),
        )

    def inverse(self):
        # type: () -> Quaternion
        return self.conj() / self.squared_norm()

    # -------------------------------------------------------------------------
    # Quaternion math helper methods
    # -------------------------------------------------------------------------

    def __mul__(self, right):
        # type: (Quaternion) -> Quaternion
        """
        Quaternion multiplication.

        Args:
            right (Quaternion):

        Returns:
            Quaternion:
        """
        return self.compose(right)

    def __neg__(self):
        # type: () -> Quaternion
        """
        Negation of all entries.

        Returns:
            Quaternion:
        """
        return self.__class__(xyz=-self.xyz, w=-self.w)

    def __add__(self, right):
        # type: (Quaternion) -> Quaternion
        """
        Quaternion addition.

        Args:
            right (Quaternion):

        Returns:
            Quaternion:
        """
        return self.__class__(xyz=self.xyz + right.xyz, w=self.w + right.w)

    def __div__(self, scalar):
        # type: (T.Scalar) -> Quaternion
        """
        Scalar division.

        Args:
            scalar (Scalar):

        Returns:
            Quaternion:
        """
        denom = sm.S.One / scalar
        return self.__class__(xyz=self.xyz * denom, w=self.w * denom)

    __truediv__ = __div__

    @classmethod
    def zero(cls):
        # type: () -> Quaternion
        """
        Construct with all zeros.

        Returns:
            Quaternion:
        """
        return cls.from_storage([0] * cls.storage_dim())

    def squared_norm(self):
        # type: () -> T.Scalar
        """
        Squared norm when considering the quaternion as 4-tuple.

        Returns:
            Scalar:
        """
        return self.xyz.dot(self.xyz) + self.w ** 2

    def conj(self):
        # type: () -> Quaternion
        """
        Quaternion conjugate.

        Returns:
            Quaternion:
        """
        return Quaternion(xyz=-self.xyz, w=self.w)

    @classmethod
    def unit_random(cls):
        # type: () -> Quaternion
        """
        Generate a random unit quaternion
        """
        u1, u2, u3 = np.random.uniform(low=0.0, high=1.0, size=(3,))
        return cls.unit_random_from_uniform_samples(u1, u2, u3, pi=np.pi)

    @classmethod
    def unit_random_from_uniform_samples(cls, u1, u2, u3, pi=sm.pi):
        # type: (T.Scalar, T.Scalar, T.Scalar, T.Scalar) -> Quaternion
        """
        Generate a random unit quaternion from three variables uniformly sampled in [0, 1].

        Reference:
            http://planning.cs.uiuc.edu/node198.html
        """
        w = sm.sqrt(u1) * sm.cos(2 * pi * u3)
        # Multiply to keep w positive to only stay on one side of double-cover
        w_sign = sm.sign(w)
        return Quaternion(
            xyz=Vector3(
                sm.sqrt(1 - u1) * sm.sin(2 * pi * u2) * w_sign,
                sm.sqrt(1 - u1) * sm.cos(2 * pi * u2) * w_sign,
                sm.sqrt(u1) * sm.sin(2 * pi * u3) * w_sign,
            ),
            w=w * w_sign,
        )
