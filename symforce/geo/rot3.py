import functools
from symforce import sympy as sm

from .base import LieGroup
from .matrix import Matrix, Z3, V3, V4
from .quaternion import Quaternion


class Rot3(LieGroup):
    """
    Group of three-dimensional orthogonal matrices with determinant +1, representing
    rotations in 3D space. Backed by a quaternion with (x, y, z, w) storage.
    """

    TANGENT_DIM = 3
    MATRIX_DIMS = (3, 3)
    STORAGE_DIM = Quaternion.STORAGE_DIM

    def __init__(self, q=None):
        """
        Construct from a unit quaternion, or identity if none provided.

        Args:
            q (Quaternion):
        """
        self.q = q if q is not None else Quaternion.identity()
        assert isinstance(self.q, Quaternion)

    # -------------------------------------------------------------------------
    # Storage concept - see symforce.ops.storage_ops
    # -------------------------------------------------------------------------

    def __repr__(self):
        return "<Rot3 {}>".format(repr(self.q))

    def to_storage(self):
        return self.q.to_storage()

    @classmethod
    def from_storage(cls, vec):
        return cls(Quaternion.from_storage(vec))

    @classmethod
    def symbolic(cls, name, **kwargs):
        return cls(Quaternion.symbolic(name, **kwargs))

    # -------------------------------------------------------------------------
    # Group concept - see symforce.ops.group_ops
    # -------------------------------------------------------------------------

    @classmethod
    def identity(cls):
        return cls(Quaternion.identity())

    def compose(self, other):
        return Rot3(self.q * other.q)

    def inverse(self):
        # NOTE(hayk): Since we have a unit quaternion, no need to call q.inv()
        # and divide by the squared norm.
        return self.__class__(self.q.conj())

    # -------------------------------------------------------------------------
    # Lie group implementation
    # -------------------------------------------------------------------------

    @classmethod
    def expmap(cls, v, epsilon=0):
        if isinstance(v, (list, tuple)):
            v = Matrix(v)
        theta_sq = v.dot(v)
        theta = sm.sqrt(theta_sq + epsilon ** 2)
        assert theta != 0, "Trying to divide by zero, provide epsilon!"
        return cls(Quaternion(xyz=sm.sin(theta / 2) / theta * v, w=sm.cos(theta / 2)))

    def logmap_signed_epsilon(self, epsilon=0):
        norm = sm.sqrt(1 + epsilon - self.q.w ** 2)
        tangent = 2 * self.q.xyz / norm * sm.acos(self.q.w - sm.sign(self.q.w) * epsilon)
        return tangent.to_storage()

    def logmap_acos_clamp_max(self, epsilon=0):
        norm = sm.sqrt(sm.Max(epsilon, 1 - self.q.w ** 2))
        tangent = 2 * self.q.xyz / norm * sm.acos(sm.Max(-1, sm.Min(1, self.q.w)))
        return tangent.to_storage()

    def logmap(self, epsilon=0):
        return self.logmap_acos_clamp_max(epsilon=epsilon)

    @classmethod
    def hat(cls, vec):
        return [[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]]

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def __mul__(self, right):
        """
        Left-multiplication. Either rotation concatenation or point transform.

        Args:
            right (SO3 or Matrix):

        Returns:
            SO3 or Matrix:
        """
        if isinstance(right, sm.Matrix):
            assert right.shape == (3, 1), right.shape
            return self.to_rotation_matrix() * right
        elif isinstance(right, Rot3):
            return self.compose(right)
        else:
            raise NotImplementedError('Unsupported type: "{}"'.format(right))

    def to_rotation_matrix(self):
        """
        A matrix representation of this element in the Euclidean space that contains it.

        Returns:
            Matrix: 3x3 rotation matrix
        """
        return self.q.to_rotation_matrix()

    @classmethod
    def from_axis_angle(cls, axis, angle):
        """
        Construct from a (normalized) axis and an angle in radians.

        Args:
            axis (Matrix): 3x1 unit vector
            angle (Scalar): rotation angle [radians]

        Returns:
            Rot3:
        """
        return cls(Quaternion.from_axis_angle(axis, angle))

    @classmethod
    def from_two_unit_vectors(cls, a, b, epsilon=0):
        """
        Return a rotation from the vector a to b. Both inputs are three-vectors that
        are expected to be normalized.

        See this reference for relevant math:
            http://lolengine.net/blog/2013/09/18/beautiful-maths-quaternion-from-vectors

        Args:
            a (Matrix): Source 3x1 unit vector
            b (Matrix): Destination 3x1 unit vector
            epsilon (Scalar): Small number to prevent singularities

        Returns:
            Rot3:
        """
        one, two = sm.S(1), sm.S(2)

        # If a.dot(b) == -1, it's a degenerate case and we need to return a 180 rotation
        # about a *different* axis. We select either the unit X or unit Y axis.
        is_valid = (sm.sign(sm.Abs(a.dot(b) + one) - epsilon) + one) / two
        is_x_vec = V3().are_parallel(a, V3(one, 0, 0), epsilon)
        non_parallel_vec = is_x_vec * V3(0, one, 0) + (one - is_x_vec) * V3(one, 0, 0)

        m = sm.sqrt(two + two * a.dot(b) + epsilon)
        return cls(
            Quaternion(
                xyz=is_valid * a.cross(b) / m + (one - is_valid) * non_parallel_vec,
                w=is_valid * m / two,
            )
        )

    def angle_between(self, other, epsilon=0):
        """
        Return the angle between this rotation and the other in radians.

        Args:
            other (Rot3):
            epsilon (Scalar): Small number to prevent singularities

        Returns:
            (Scalar):
        """
        return Matrix(self.local_coordinates(other, epsilon=epsilon)).norm()
