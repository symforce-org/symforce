# mypy: disallow-untyped-defs

import numpy as np

from symforce import sympy as sm
from symforce import types as T

from .base import LieGroup
from .matrix import Matrix
from .matrix import V3
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
        # type: (Quaternion) -> None
        """
        Construct from a unit quaternion, or identity if none provided.
        """
        self.q = q if q is not None else Quaternion.identity()
        assert isinstance(self.q, Quaternion)

    # -------------------------------------------------------------------------
    # Storage concept - see symforce.ops.storage_ops
    # -------------------------------------------------------------------------

    def __repr__(self):
        # type: () -> str
        return "<Rot3 {}>".format(repr(self.q))

    def to_storage(self):
        # type: () -> T.List[T.Scalar]
        return self.q.to_storage()

    @classmethod
    def from_storage(cls, vec):
        # type: (T.List) -> Rot3
        return cls(Quaternion.from_storage(vec))

    @classmethod
    def symbolic(cls, name, **kwargs):
        # type: (str, T.Any) -> Rot3
        return cls(Quaternion.symbolic(name, **kwargs))

    # -------------------------------------------------------------------------
    # Group concept - see symforce.ops.group_ops
    # -------------------------------------------------------------------------

    @classmethod
    def identity(cls):
        # type: () -> Rot3
        return cls(Quaternion.identity())

    def compose(self, other):
        # type: (Rot3) -> Rot3
        return Rot3(self.q * other.q)

    def inverse(self):
        # type: () -> Rot3
        # NOTE(hayk): Since we have a unit quaternion, no need to call q.inv()
        # and divide by the squared norm.
        return self.__class__(self.q.conj())

    # -------------------------------------------------------------------------
    # Lie group implementation
    # -------------------------------------------------------------------------

    @classmethod
    def expmap(cls, v, epsilon=0):
        # type: (T.List[T.Scalar], T.Scalar) -> Rot3
        vm = Matrix(v)
        theta_sq = vm.dot(vm)
        theta = sm.sqrt(theta_sq + epsilon ** 2)
        assert theta != 0, "Trying to divide by zero, provide epsilon!"
        return cls(Quaternion(xyz=sm.sin(theta / 2) / theta * vm, w=sm.cos(theta / 2)))

    def logmap_signed_epsilon(self, epsilon=0):
        # type: (float) -> T.List[T.Scalar]
        """
        Implementation of logmap that uses epsilon with the sign function to avoid NaN.
        """
        norm = sm.sqrt(1 + epsilon - self.q.w ** 2)
        tangent = 2 * self.q.xyz / norm * sm.acos(self.q.w - sm.sign(self.q.w) * epsilon)
        return tangent.to_storage()

    def logmap_acos_clamp_max(self, epsilon=0):
        # type: (float) -> T.List[T.Scalar]
        """
        Implementation of logmap that uses epsilon with the Max and Min functions to avoid NaN.
        """
        norm = sm.sqrt(sm.Max(epsilon, 1 - self.q.w ** 2))
        tangent = 2 * self.q.xyz / norm * sm.acos(sm.Max(-1, sm.Min(1, self.q.w)))
        return tangent.to_storage()

    def logmap(self, epsilon=0):
        # type: (float) -> T.List[T.Scalar]
        return self.logmap_acos_clamp_max(epsilon=epsilon)

    @classmethod
    def hat(cls, vec):
        # type: (T.List[T.Scalar]) -> T.List[T.List[T.Scalar]]
        return [[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]]

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def __mul__(self, right):
        # type: (T.Union[Matrix, Rot3]) -> T.Any
        """
        Left-multiplication. Either rotation concatenation or point transform.
        """
        if isinstance(right, sm.Matrix):
            assert right.shape == (3, 1), right.shape
            return self.to_rotation_matrix() * right
        elif isinstance(right, Rot3):
            return self.compose(right)
        else:
            raise NotImplementedError('Unsupported type: "{}"'.format(right))

    def to_rotation_matrix(self):
        # type: () -> Matrix
        """
        3x3 rotation matrix
        """
        return self.q.to_rotation_matrix()

    @classmethod
    def from_axis_angle(cls, axis, angle):
        # type: (Matrix, T.Scalar) -> Rot3
        """
        Construct from a (normalized) axis as a 3-vector and an angle in radians.
        """
        return cls(Quaternion.from_axis_angle(axis, angle))

    @classmethod
    def from_two_unit_vectors(cls, a, b, epsilon=0):
        # type: (Matrix, Matrix, T.Scalar) -> Rot3
        """
        Return a rotation that transforms a to b. Both inputs are three-vectors that
        are expected to be normalized.

        Reference:
            http://lolengine.net/blog/2013/09/18/beautiful-maths-quaternion-from-vectors
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
        # type: (Rot3, T.Scalar) -> T.Scalar
        """
        Return the angle between this rotation and the other in radians.
        """
        return Matrix(self.local_coordinates(other, epsilon=epsilon)).norm()

    @classmethod
    def random(cls):
        # type: () -> Rot3
        """
        Generate a random element of SO3.
        """
        u1, u2, u3 = np.random.uniform(low=0.0, high=1.0, size=(3,))
        return cls.random_from_uniform_samples(u1, u2, u3, pi=np.pi)

    @classmethod
    def random_from_uniform_samples(cls, u1, u2, u3, pi=sm.pi):
        # type: (T.Scalar, T.Scalar, T.Scalar, T.Scalar) -> Rot3
        """
        Generate a random element of SO3 from three variables uniformly sampled in [0, 1].

        Reference:
            http://planning.cs.uiuc.edu/node198.html
        """
        w = sm.sqrt(u1) * sm.cos(2 * pi * u3)
        # Multiply to keep w positive to only stay on one side of double-cover
        w_sign = sm.sign(w)
        return cls(
            q=Quaternion(
                xyz=V3(
                    sm.sqrt(1 - u1) * sm.sin(2 * pi * u2) * w_sign,
                    sm.sqrt(1 - u1) * sm.cos(2 * pi * u2) * w_sign,
                    sm.sqrt(u1) * sm.sin(2 * pi * u3) * w_sign,
                ),
                w=w * w_sign,
            )
        )
