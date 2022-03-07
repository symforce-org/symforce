# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

from symforce.ops.interfaces.lie_group import LieGroup
from symforce import typing as T
from symforce import sympy as sm

from .matrix import Matrix
from .matrix import Matrix13
from .matrix import Matrix21
from .matrix import Matrix33
from .matrix import Vector2
from .rot2 import Rot2

from .pose2 import Pose2


class Pose2_SE2(Pose2):
    """
    Group of two-dimensional rigid body transformations - SE(2).

    The storage space is a complex (real, imag) for rotation followed by a position (x, y).

    The tangent space is two elements for translation in the rotated frame followed by one angle
    for rotation. This means we interpolate the translation in the tangent of the rotating frame
    for lie operations. This can be useful but is more expensive than R2 x SO(2) for often no
    benefit.
    """

    # -------------------------------------------------------------------------
    # Lie group implementation
    # -------------------------------------------------------------------------

    @classmethod
    def from_tangent(cls, v: T.Sequence[T.Scalar], epsilon: T.Scalar = 0) -> Pose2_SE2:
        theta = v[2]
        R = Rot2.from_tangent([theta], epsilon=epsilon)

        a = (R.z.imag + epsilon * sm.sign_no_zero(R.z.imag)) / (
            theta + epsilon * sm.sign_no_zero(theta)
        )
        b = (1 - R.z.real) / (theta + epsilon * sm.sign_no_zero(theta))

        t = Vector2(a * v[0] - b * v[1], b * v[0] + a * v[1])
        return cls(R, t)

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

    def retract(self, vec: T.Sequence[T.Scalar], epsilon: T.Scalar = 0) -> Pose2_SE2:
        return LieGroup.retract(self, vec, epsilon)

    def local_coordinates(self, b: Pose2_SE2, epsilon: T.Scalar = 0) -> T.List[T.Scalar]:
        return LieGroup.local_coordinates(self, b, epsilon)

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    @classmethod
    def hat(cls, vec: T.List[T.Scalar]) -> Matrix33:
        t_tangent = [vec[0], vec[1]]
        R_tangent = [vec[2]]
        top_left = Rot2.hat(R_tangent)
        top_right = Matrix21(t_tangent)
        bottom = Matrix13.zero()
        return T.cast(Matrix33, top_left.row_join(top_right).col_join(bottom))
