from __future__ import division

from symforce import sympy as sm

from .base import Group
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

    STORAGE_DIM = 4

    def __init__(self, xyz, w):
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
        return self.xyz[0]

    @property
    def y(self):
        return self.xyz[1]

    @property
    def z(self):
        return self.xyz[2]

    # -------------------------------------------------------------------------
    # Storage concept - see symforce.ops.storage_ops
    # -------------------------------------------------------------------------

    def __repr__(self):
        return "<Q xyzw=[{}, {}, {}, {}]>".format(
            repr(self.x), repr(self.y), repr(self.z), repr(self.w)
        )

    def to_storage(self):
        return [self.x, self.y, self.z, self.w]

    @classmethod
    def from_storage(cls, vec):
        assert len(vec) == cls.STORAGE_DIM
        return cls(xyz=Matrix(vec[0:3]), w=vec[3])

    @classmethod
    def symbolic(cls, name, **kwargs):
        return cls.from_storage(
            [sm.Symbol("{}_{}".format(name, v), **kwargs) for v in ["x", "y", "z", "w"]]
        )

    # -------------------------------------------------------------------------
    # Group concept - see symforce.ops.group_ops
    # -------------------------------------------------------------------------

    @classmethod
    def identity(cls):
        return cls(xyz=Vector3(0, 0, 0), w=1)

    def compose(self, other):
        return self.__class__(
            xyz=self.w * other.xyz + other.w * self.xyz + self.xyz.cross(other.xyz),
            w=self.w * other.w - self.xyz.dot(other.xyz),
        )

    def inverse(self):
        return self.conj() / self.squared_norm()

    # -------------------------------------------------------------------------
    # Quaternion math helper methods
    # -------------------------------------------------------------------------

    def __mul__(self, right):
        """
        Quaternion multiplication.

        Args:
            right (Quaternion):

        Returns:
            Quaternion:
        """
        return self.compose(right)

    def __neg__(self):
        """
        Negation of all entries.

        Returns:
            Quaternion:
        """
        return self.__class__(xyz=-self.xyz, w=-self.w)

    def __add__(self, right):
        """
        Quaternion addition.

        Args:
            right (Quaternion):

        Returns:
            Quaternion:
        """
        return self.__class__(xyz=self.xyz + right.xyz, w=self.w + right.w)

    def __truediv__(self, scalar):
        """
        Scalar division.

        Args:
            scalar (Scalar):
        
        Returns:
            Quaternion:
        """
        denom = sm.S.One / scalar
        return self.__class__(xyz=self.xyz * denom, w=self.w * denom)

    @classmethod
    def zero(cls):
        """
        Construct with all zeros.

        Returns:
            Quaternion:
        """
        return cls.from_storage([0] * cls.STORAGE_DIM)

    def squared_norm(self):
        """
        Squared norm when considering the quaternion as 4-tuple.

        Returns:
            Quaternion:
        """
        return self.xyz.dot(self.xyz) + self.w ** 2

    def conj(self):
        """
        Quaternion conjugate.
        
        Returns:
            Quaternion:
        """
        return Quaternion(xyz=-self.xyz, w=self.w)

    # -------------------------------------------------------------------------
    # Rotation-specific helper methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_axis_angle(cls, axis, angle):
        """
        Construct from a (normalized) axis and an angle in radians.

        Args:
            axis (Matrix): 3x1 unit vector
            angle (float): rotation angle [radians]

        Returns:
            Quaternion:
        """
        return Quaternion(xyz=axis * sm.sin(angle / 2), w=sm.cos(angle / 2))

    def yaw_angle(self, epsilon=0):
        """
        Compute the yaw angle of this as the projection of the rotated X axis
        relative to the world X axis on the XY plane.

        Args:
            epsilon (Scalar): Small number to optionally prevent degeneracy.

        Returns:
            Scalar: World yaw euler angle
        """
        x_W = self * Quaternion(w=0, xyz=Vector3([1, 0, 0])) * self.conj()
        return sm.atan2_safe(x_W.y, x_W.x, epsilon=epsilon)

    def to_rotation_matrix(self):
        """
        Converts to a rotation matrix, assuming this is a unit quaternion.

        Returns:
            Matrix: 3x3 matrix representing this quaternion as a rotation
        """
        return Matrix(
            [
                [
                    1 - 2 * self.xyz[1] ** 2 - 2 * self.xyz[2] ** 2,
                    2 * self.xyz[0] * self.xyz[1] - 2 * self.xyz[2] * self.w,
                    2 * self.xyz[0] * self.xyz[2] + 2 * self.xyz[1] * self.w,
                ],
                [
                    2 * self.xyz[0] * self.xyz[1] + 2 * self.xyz[2] * self.w,
                    1 - 2 * self.xyz[0] ** 2 - 2 * self.xyz[2] ** 2,
                    2 * self.xyz[1] * self.xyz[2] - 2 * self.xyz[0] * self.w,
                ],
                [
                    2 * self.xyz[0] * self.xyz[2] - 2 * self.xyz[1] * self.w,
                    2 * self.xyz[1] * self.xyz[2] + 2 * self.xyz[0] * self.w,
                    1 - 2 * self.xyz[0] ** 2 - 2 * self.xyz[1] ** 2,
                ],
            ]
        )

    @classmethod
    def from_rotation_matrix(cls, R, epsilon=0):
        """
        Creates a Quaternion from a rotation matrix

        Implementation copied from Eigen::Quaternion
        NOTE: this is only stable if trace > 0

        Args:
            R (Matrix): 3x3 rotation matrix
            epsilon (Scalar): Small number to optionally prevent degeneracy.

        Returns:
            Quaternion:
        """

        trace = R[0, 0] + R[1, 1] + R[2, 2] + epsilon
        r = sm.sqrt(1 + trace)
        w = r / 2.0
        s = 1.0 / (2 * r)
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
        return cls(xyz=Vector3(x, y, z), w=w)
