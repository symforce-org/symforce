# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import symforce.internal.symbolic as sf
from symforce import typing as T
from symforce.ops.interfaces import LieGroup

from .matrix import V3
from .matrix import Matrix
from .matrix import Matrix33
from .matrix import Matrix34
from .matrix import Matrix43
from .matrix import Vector3
from .matrix import Vector4
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
        return self.__class__(self.q * other.q)

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
    def from_tangent(cls, v: T.Sequence[T.Scalar], epsilon: T.Scalar = sf.epsilon()) -> Rot3:
        vm = Matrix(v)
        theta_sq = vm.squared_norm()
        theta = sf.sqrt(theta_sq + epsilon ** 2)
        assert theta != 0, "Trying to divide by zero, provide epsilon!"
        return cls(Quaternion(xyz=sf.sin(theta / 2) / theta * vm, w=sf.cos(theta / 2)))

    def logmap_acos_clamp_max(self, epsilon: T.Scalar = sf.epsilon()) -> T.List[T.Scalar]:
        """
        Implementation of logmap that uses epsilon with the Min function to
        avoid the singularity in the sqrt at w == 1

        Also flips the sign of the quaternion of w is negative, which makes sure
        that the resulting tangent vector has norm <= pi
        """
        w_positive = sf.Abs(self.q.w)
        w_safe = sf.Min(1 - epsilon, w_positive)
        xyz_w_positive = self.q.xyz * sf.sign_no_zero(self.q.w)
        norm = sf.sqrt(1 - w_safe ** 2)
        tangent = 2 * xyz_w_positive / norm * sf.acos(w_safe)
        return tangent.to_storage()

    def to_tangent(self, epsilon: T.Scalar = sf.epsilon()) -> T.List[T.Scalar]:
        return self.logmap_acos_clamp_max(epsilon=epsilon)

    @classmethod
    def hat(cls, vec: T.Sequence[T.Scalar]) -> Matrix33:
        return Matrix33([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])

    def storage_D_tangent(self) -> Matrix43:
        """
        Note: generated from symforce/notebooks/storage_D_tangent.ipynb
        """
        return (
            sf.S.One
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
    def __mul__(self, right: Vector3) -> Vector3:  # pragma: no cover
        pass

    @T.overload
    def __mul__(self, right: Rot3) -> Rot3:  # pragma: no cover
        pass

    def __mul__(self, right: T.Union[Vector3, Rot3]) -> T.Union[Vector3, Rot3]:
        """
        Left-multiplication. Either rotation concatenation or point transform.
        """
        if isinstance(right, Vector3):
            return T.cast(Vector3, self.to_rotation_matrix() * right)
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
    def from_rotation_matrix(cls, R: Matrix33, epsilon: T.Scalar = sf.epsilon()) -> Rot3:
        """
        Construct from a rotation matrix.
        """

        # This implementation is based off of that found in Eigen's Quaternion.h, found at
        # https://gitlab.com/libeigen/eigen/-/blob/4091f6b2/Eigen/src/Geometry/Quaternion.h#L814
        # which gets the algorithm from
        # "Quaternion Calculus and Fast Animation",
        # Ken Shoemake, 1987 SIGGRAPH course notes
        #
        # For a helpful discussion of the problem, see the paper
        # http://www.iri.upc.edu/files/scidoc/2068-Accurate-Computation-of-Quaternions-from-Rotation-Matrices.pdf
        #
        # Since there is no simple expression we can use to calculate the corresponding
        # quaternion q from R which is numerically stable everywhere, we arm ourselves with
        # four different expressions so that we always have one which is stable for any given R.
        #
        # Each expression corresponds to one component of q (q_0, q_1, q_2, w) that is appropriate
        # to use when that component is not close to zero.
        #
        # We choose the expression whose corresponding component has the largest magnitude as that
        # expression is guaranteed to not divide by zero (or a similarly small number).
        #
        # We select our desired expression by evaluating all of them (modifying the ones we're
        # not using to avoid dividing by zero), multiplying the one we want to use by 1 and
        # the ones we don't want by 0, and adding them together.
        #
        # The expressions corresponding to the spatial components of q are sufficiently similar
        # that we can express them with a single function, from_rotation_matrix_qi_not_0. The
        # w component's expression is returned by from_rotation_matrix_w_not_0

        # Useful identities for understanding the expressions:
        # trace(R) = 1 + 2cos(theta) = 4w^2 - 1
        # If q_0 = x, q_1 = y, q_2 = z, and
        # i,j,k is an even permutation of 0,1,2, then
        #   R[j, i] - R[i, j] = 4 * w * q_k
        #   R[j, i] + R[i, j] = 4 * q_i * q_j
        # R[i, i] = 2(w^2 + q_i^2)

        def from_rotation_matrix_w_not_0(R: Matrix33, valid_input: T.Scalar) -> Quaternion:
            """
            If valid_input is 0, returns the zero quaternion.
            If valid_input is 1, returns a quaternion q such that qv(q^-1) = Rv for all vectors v
            Numerical performance is best when |w| is not close to 0
            Preconditions:
                valid_input must be either exactly 0 or exactly 1
                If trace(R) = -1 (i.e., w = 0), then valid_input must be 0
            """
            # abs_2w is |2w| = |2cos(theta/2)|.
            # We add not(valid_input) to ensure abs_2w is non-zero to avoid dividing by zero
            # in the case where we wish to return the zero quaternion
            abs_2w = sf.sqrt(sf.Max(1 + R[0, 0] + R[1, 1] + R[2, 2], 0)) + sf.logical_not(
                valid_input, unsafe=True
            )
            w = abs_2w / 2
            inv_abs_4w = 1 / (2 * abs_2w)
            x = (R[2, 1] - R[1, 2]) * inv_abs_4w
            y = (R[0, 2] - R[2, 0]) * inv_abs_4w
            z = (R[1, 0] - R[0, 1]) * inv_abs_4w
            *xyz, w = valid_input * Vector4(x, y, z, w)
            return Quaternion(xyz=V3(*xyz), w=w)

        def from_rotation_matrix_qi_not_0(R: Matrix33, i: int, valid_input: T.Scalar) -> Quaternion:
            """
            If valid_input is 0, returns the zero quaternion.
            If valid_input is 1, returns a quaternion q such that qv(q^-1) = Rv for all vectors v
            Numerical performance is best when |q_i| is not close to 0
            Preconditions:
                valid_input must be either exactly 0 or exactly 1
                If 2*R[i, i] - trace(R) = -1 (i.e., q_i = 0), then valid_input must be 0
            """
            j = (i + 1) % 3
            k = (i + 2) % 3
            components = [0] * 3
            # abs_2q_i = |2q_i| = |2sin(theta/2)a_i| where a is the axis of rotation
            # We add not(valid_input) to ensure abs_2q_i is non-zero to avoid dividing by zero
            # in the case where we wish to return the zero quaternion
            abs_2q_i = sf.sqrt(sf.Max(1 + R[i, i] - R[j, j] - R[k, k], 0)) + sf.logical_not(
                valid_input, unsafe=True
            )
            components[i] = abs_2q_i / 2
            inv_abs_4q_i = 1 / (2 * abs_2q_i)
            w = (R[k, j] - R[j, k]) * inv_abs_4q_i
            components[j] = (R[j, i] + R[i, j]) * inv_abs_4q_i
            components[k] = (R[k, i] + R[i, k]) * inv_abs_4q_i
            *xyz, w = valid_input * Vector4(*components, w)
            return Quaternion(xyz=V3(*xyz), w=w)

        # Relevant math needed to justify the next line:
        #             R00 = 2(w^2 + x^2) - 1
        #             R11 = 2(w^2 + y^2) - 1
        #             R22 = 2(w^2 + z^2) - 1
        # R00 + R11 + R22 = 4w^2 - 1
        # So max(R00, R11, R22, R00 + R11 + R22) = max(x^2, y^2, z^2, w^2)
        # x^2 + y^2 + z^2 + w^2 = 1  =>  max(x^2, y^2, z^2, w^2) >= 1/4
        # This is all to say that if use_branch[i] = 1, then it's safe to use q_i's expression
        use_branch = sf.argmax_onehot([R[0, 0], R[1, 1], R[2, 2], R[0, 0] + R[1, 1] + R[2, 2]])
        q = from_rotation_matrix_w_not_0(R, use_branch[3])
        for i in range(0, 3):
            q += from_rotation_matrix_qi_not_0(R, i, use_branch[i])
        return cls(q)

    def to_yaw_pitch_roll(
        self, epsilon: T.Scalar = sf.epsilon()
    ) -> T.Tuple[T.Scalar, T.Scalar, T.Scalar]:
        """
        Compute the yaw, pitch, and roll Euler angles in radians of this rotation

        Returns:
            Scalar: Yaw angle [radians]
            Scalar: Pitch angle [radians]
            Scalar: Roll angle [radians]
        """
        y = sf.atan2(
            2 * self.q.x * self.q.y + 2 * self.q.w * self.q.z,
            self.q.x * self.q.x + self.q.w * self.q.w - self.q.z * self.q.z - self.q.y * self.q.y,
            epsilon,
        )
        p = -sf.asin_safe(2 * self.q.x * self.q.z - 2 * self.q.w * self.q.y, epsilon)
        r = sf.atan2(
            2 * self.q.y * self.q.z + 2 * self.q.w * self.q.x,
            self.q.z * self.q.z - self.q.y * self.q.y - self.q.x * self.q.x + self.q.w * self.q.w,
            epsilon,
        )
        return y, p, r

    @classmethod
    def from_yaw_pitch_roll(
        cls, yaw: T.Scalar = 0, pitch: T.Scalar = 0, roll: T.Scalar = 0
    ) -> Rot3:
        """
        Construct from yaw, pitch, and roll Euler angles in radians
        """
        return (
            Rot3.from_angle_axis(yaw, V3(0, 0, 1))
            * Rot3.from_angle_axis(pitch, V3(0, 1, 0))
            * Rot3.from_angle_axis(roll, V3(1, 0, 0))
        )

    @classmethod
    def from_angle_axis(cls, angle: T.Scalar, axis: Vector3) -> Rot3:
        """
        Construct from an angle in radians and a (normalized) axis as a 3-vector.
        """
        return cls(Quaternion(xyz=axis * sf.sin(angle / 2), w=sf.cos(angle / 2)))

    @classmethod
    def from_two_unit_vectors(
        cls, a: Vector3, b: Vector3, epsilon: T.Scalar = sf.epsilon()
    ) -> Rot3:
        """
        Return a rotation that transforms a to b. Both inputs are three-vectors that
        are expected to be normalized.

        Reference:
            http://lolengine.net/blog/2013/09/18/beautiful-maths-quaternion-from-vectors
        """
        one, two = sf.S(1), sf.S(2)

        # If a.dot(b) == -1, it's a degenerate case and we need to return a 180 rotation
        # about a *different* axis. We select either the unit X or unit Y axis.
        is_valid = (sf.sign(sf.Abs(a.dot(b) + one) - epsilon) + one) / two
        is_x_vec = V3.are_parallel(a, V3(one, 0, 0), epsilon)
        non_parallel_vec = is_x_vec * V3(0, one, 0) + (one - is_x_vec) * V3(one, 0, 0)

        m = sf.sqrt(two + two * a.dot(b) + epsilon)
        return cls(
            Quaternion(
                xyz=is_valid * a.cross(b) / m + (one - is_valid) * non_parallel_vec,
                w=is_valid * m / two,
            )
        )

    def angle_between(self, other: Rot3, epsilon: T.Scalar = sf.epsilon()) -> T.Scalar:
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
        cls, u1: T.Scalar, u2: T.Scalar, u3: T.Scalar, pi: T.Scalar = sf.pi
    ) -> Rot3:
        """
        Generate a random element of SO3 from three variables uniformly sampled in [0, 1].
        """
        return cls(Quaternion.unit_random_from_uniform_samples(u1, u2, u3, pi))
