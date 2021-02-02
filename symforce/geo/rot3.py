from __future__ import annotations

import numpy as np

from symforce.ops.interfaces import LieGroup
from symforce import sympy as sm
from symforce import types as T

from .matrix import Matrix
from .matrix import Matrix31
from .matrix import Matrix33
from .matrix import Matrix34
from .matrix import Matrix43
from .matrix import V3
from .quaternion import Quaternion


class Rot3(LieGroup):
    """
    Group of three-dimensional orthogonal matrices with determinant +1, representing
    rotations in 3D space. Backed by a quaternion with (x, y, z, w) storage.
    """

    def __init__(self, q: Quaternion = None) -> None:
        """
        Construct from a unit quaternion, or identity if none provided.
        """
        self.q = q if q is not None else Quaternion.identity()
        assert isinstance(self.q, Quaternion)

    # -------------------------------------------------------------------------
    # Storage concept - see symforce.ops.storage_ops
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return "<Rot3 {}>".format(repr(self.q))

    @classmethod
    def storage_dim(cls) -> int:
        return Quaternion.storage_dim()

    def to_storage(self) -> T.List[T.Scalar]:
        return self.q.to_storage()

    @classmethod
    def from_storage(cls, vec: T.Sequence[T.Scalar]) -> Rot3:
        return cls(Quaternion.from_storage(vec))

    @classmethod
    def symbolic(cls, name: str, **kwargs: T.Any) -> Rot3:
        return cls(Quaternion.symbolic(name, **kwargs))

    # -------------------------------------------------------------------------
    # Group concept - see symforce.ops.group_ops
    # -------------------------------------------------------------------------

    @classmethod
    def identity(cls) -> Rot3:
        return cls(Quaternion.identity())

    def compose(self, other: Rot3) -> Rot3:
        return Rot3(self.q * other.q)

    def inverse(self) -> Rot3:
        # NOTE(hayk): Since we have a unit quaternion, no need to call q.inv()
        # and divide by the squared norm.
        return self.__class__(self.q.conj())

    # -------------------------------------------------------------------------
    # Lie group implementation
    # -------------------------------------------------------------------------

    @classmethod
    def tangent_dim(cls) -> int:
        return 3

    @classmethod
    def from_tangent(cls, v: T.Sequence[T.Scalar], epsilon: T.Scalar = 0) -> Rot3:
        vm = Matrix(v)
        theta_sq = vm.squared_norm()
        theta = sm.sqrt(theta_sq + epsilon ** 2)
        assert theta != 0, "Trying to divide by zero, provide epsilon!"
        return cls(Quaternion(xyz=sm.sin(theta / 2) / theta * vm, w=sm.cos(theta / 2)))

    def logmap_acos_clamp_max(self, epsilon: T.Scalar = 0) -> T.List[T.Scalar]:
        """
        Implementation of logmap that uses epsilon with the Min function to
        avoid the singularity in the sqrt at w == 1

        Also flips the sign of the quaternion of w is negative, which makes sure
        that the resulting tangent vector has norm <= pi

        See discussion:
        ***REMOVED***
        ***REMOVED***
        ***REMOVED***
        As well as symforce/notebooks/epsilon-sandbox.ipynb
        """
        w_positive = sm.Abs(self.q.w)
        w_safe = sm.Min(1 - epsilon, w_positive)
        xyz_w_positive = self.q.xyz * sm.sign_no_zero(self.q.w)
        norm = sm.sqrt(1 - w_safe ** 2)
        tangent = 2 * xyz_w_positive / norm * sm.acos(w_safe)
        return tangent.to_storage()

    def to_tangent(self, epsilon: T.Scalar = 0) -> T.List[T.Scalar]:
        return self.logmap_acos_clamp_max(epsilon=epsilon)

    @classmethod
    def hat(cls, vec: T.Sequence[T.Scalar]) -> Matrix33:
        return Matrix33([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])

    def storage_D_tangent(self) -> Matrix43:
        """
        Note: generated from symforce/notebooks/storage_D_tangent.ipynb
        """
        return (
            sm.S.One
            / 2
            * Matrix(
                [
                    [self.q.w, -self.q.z, self.q.y],
                    [self.q.z, self.q.w, -self.q.x],
                    [-self.q.y, self.q.x, self.q.w],
                    [-self.q.x, -self.q.y, -self.q.z],
                ]
            )
        )

    def tangent_D_storage(self) -> Matrix34:
        """
        Note: generated from symforce/notebooks/tangent_D_storage.ipynb
        """
        return 4 * T.cast(Matrix34, self.storage_D_tangent().T)

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    @T.overload
    def __mul__(self, right: Matrix31) -> Matrix31:  # pragma: no cover
        pass

    @T.overload
    def __mul__(self, right: Rot3) -> Rot3:  # pragma: no cover
        pass

    def __mul__(self, right: T.Union[Matrix31, Rot3]) -> T.Union[Matrix31, Rot3]:
        """
        Left-multiplication. Either rotation concatenation or point transform.
        """
        if isinstance(right, V3):
            return T.cast(Matrix31, self.to_rotation_matrix() * right)
        elif isinstance(right, Rot3):
            return self.compose(right)
        else:
            raise NotImplementedError(f'Unsupported type: "{right}"')

    def to_rotation_matrix(self) -> Matrix33:
        """
        Converts to a rotation matrix
        """
        return Matrix33(
            [
                [
                    1 - 2 * self.q.y ** 2 - 2 * self.q.z ** 2,
                    2 * self.q.x * self.q.y - 2 * self.q.z * self.q.w,
                    2 * self.q.x * self.q.z + 2 * self.q.y * self.q.w,
                ],
                [
                    2 * self.q.x * self.q.y + 2 * self.q.z * self.q.w,
                    1 - 2 * self.q.x ** 2 - 2 * self.q.z ** 2,
                    2 * self.q.y * self.q.z - 2 * self.q.x * self.q.w,
                ],
                [
                    2 * self.q.x * self.q.z - 2 * self.q.y * self.q.w,
                    2 * self.q.y * self.q.z + 2 * self.q.x * self.q.w,
                    1 - 2 * self.q.x ** 2 - 2 * self.q.y ** 2,
                ],
            ]
        )

    @classmethod
    def from_rotation_matrix(cls, R: Matrix33, epsilon: T.Scalar = 0) -> Rot3:
        """
        Construct from a rotation matrix.

        Source:
            http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/christian.htm
        """
        w = sm.sqrt(sm.Max(epsilon ** 2, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
        x = sm.sqrt(sm.Max(epsilon ** 2, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
        y = sm.sqrt(sm.Max(epsilon ** 2, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
        z = sm.sqrt(sm.Max(epsilon ** 2, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
        x = sm.copysign_no_zero(x, R[2, 1] - R[1, 2])
        y = sm.copysign_no_zero(y, R[0, 2] - R[2, 0])
        z = sm.copysign_no_zero(z, R[1, 0] - R[0, 1])
        return cls(Quaternion(xyz=V3(x, y, z), w=w))

    def to_euler_ypr(self, epsilon: T.Scalar = 0) -> T.Tuple[T.Scalar, T.Scalar, T.Scalar]:
        """
        Compute the yaw, pitch, and roll Euler angles in radians of this rotation

        Returns:
            Scalar: Yaw angle [radians]
            Scalar: Pitch angle [radians]
            Scalar: Roll angle [radians]
        """
        y = sm.atan2_safe(
            2 * self.q.x * self.q.y + 2 * self.q.w * self.q.z,
            self.q.x * self.q.x + self.q.w * self.q.w - self.q.z * self.q.z - self.q.y * self.q.y,
            epsilon,
        )
        p = -sm.asin_safe(2 * self.q.x * self.q.z - 2 * self.q.w * self.q.y, epsilon)
        r = sm.atan2_safe(
            2 * self.q.y * self.q.z + 2 * self.q.w * self.q.x,
            self.q.z * self.q.z - self.q.y * self.q.y - self.q.x * self.q.x + self.q.w * self.q.w,
            epsilon,
        )
        return y, p, r

    @classmethod
    def from_euler_ypr(cls, yaw: T.Scalar, pitch: T.Scalar, roll: T.Scalar) -> Rot3:
        """
        Construct from yaw, pitch, and roll Euler angles in radians
        """
        return (
            Rot3.from_axis_angle(V3(0, 0, 1), yaw)
            * Rot3.from_axis_angle(V3(0, 1, 0), pitch)
            * Rot3.from_axis_angle(V3(1, 0, 0), roll)
        )

    @classmethod
    def from_axis_angle(cls, axis: Matrix31, angle: T.Scalar) -> Rot3:
        """
        Construct from a (normalized) axis as a 3-vector and an angle in radians.
        """
        return cls(Quaternion(xyz=axis * sm.sin(angle / 2), w=sm.cos(angle / 2)))

    @classmethod
    def from_two_unit_vectors(cls, a: Matrix31, b: Matrix31, epsilon: T.Scalar = 0) -> Rot3:
        """
        Return a rotation that transforms a to b. Both inputs are three-vectors that
        are expected to be normalized.

        Reference:
            http://lolengine.net/blog/2013/09/18/beautiful-maths-quaternion-from-vectors
        """
        one, two = sm.S(1), sm.S(2)

        # If a.dot(b) == -1, it's a degenerate case and we need to return a 180 rotation
        # about a *different* axis. We select either the unit X or unit Y axis.
        is_valid = (sm.sign(sm.Abs(a.dot(b)[0, 0] + one) - epsilon) + one) / two
        is_x_vec = V3.are_parallel(a, V3(one, 0, 0), epsilon)
        non_parallel_vec = is_x_vec * V3(0, one, 0) + (one - is_x_vec) * V3(one, 0, 0)

        m = sm.sqrt(two + two * a.dot(b)[0, 0] + epsilon)
        return cls(
            Quaternion(
                xyz=is_valid * a.cross(b) / m + (one - is_valid) * non_parallel_vec,
                w=is_valid * m / two,
            )
        )

    def angle_between(self, other: Rot3, epsilon: T.Scalar = 0) -> T.Scalar:
        """
        Return the angle between this rotation and the other in radians.
        """
        return Matrix(self.local_coordinates(other, epsilon=epsilon)).norm()

    @classmethod
    def random(cls) -> Rot3:
        """
        Generate a random element of SO3.
        """
        return cls(Quaternion.unit_random())

    @classmethod
    def random_from_uniform_samples(
        cls, u1: T.Scalar, u2: T.Scalar, u3: T.Scalar, pi: T.Scalar = sm.pi
    ) -> Rot3:
        """
        Generate a random element of SO3 from three variables uniformly sampled in [0, 1].
        """
        return cls(Quaternion.unit_random_from_uniform_samples(u1, u2, u3, pi))
