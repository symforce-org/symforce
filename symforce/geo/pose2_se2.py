# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import symforce.internal.symbolic as sf
from symforce import typing as T
from symforce.ops.interfaces.lie_group import LieGroup

from .matrix import Matrix
from .matrix import Matrix13
from .matrix import Matrix21
from .matrix import Matrix33
from .matrix import Vector2
from .pose2 import Pose2
from .rot2 import Rot2


class Pose2_SE2(Pose2):
    """
    Group of two-dimensional rigid body transformations - SE(2).

    There is no generated runtime analogue of this class in the `sym` package, which means you
    cannot use it as an input or output of generated functions or as a variable in an optimized
    Values.  This is intentional - in general, you should use the Pose2 class instead of this one,
    because the generated expressions will be significantly more efficient.  If you are sure that
    you need the different behavior of this class, it's here for reference or for use in symbolic
    expressions.

    The storage space is a complex (real, imag) for rotation followed by a position (x, y).

    The tangent space is one angle for rotation followed by two elements for translation in the
    rotated frame. This means we interpolate the translation in the tangent of the rotating frame
    for lie operations. This can be useful but is more expensive than SO(2) x R2 for often no
    benefit.
    """

    # -------------------------------------------------------------------------
    # Lie group implementation
    # -------------------------------------------------------------------------

    @classmethod
    def from_tangent(cls, v: T.Sequence[T.Scalar], epsilon: T.Scalar = sf.epsilon()) -> Pose2_SE2:
        theta = v[0]
        R = Rot2.from_tangent([theta], epsilon=epsilon)

        a = (R.z.imag + epsilon * sf.sign_no_zero(R.z.imag)) / (
            theta + epsilon * sf.sign_no_zero(theta)
        )
        b = (1 - R.z.real) / (theta + epsilon * sf.sign_no_zero(theta))

        t = Vector2(a * v[1] - b * v[2], b * v[1] + a * v[2])
        return cls(R, t)

    def to_tangent(self, epsilon: T.Scalar = sf.epsilon()) -> T.List[T.Scalar]:
        # This uses atan2, so the resulting theta is between -pi and pi
        theta = self.R.to_tangent(epsilon=epsilon)[0]

        halftheta = 0.5 * (theta + sf.sign_no_zero(theta) * epsilon)
        a = (
            halftheta
            * (1 + self.R.z.real)
            / (self.R.z.imag + sf.sign_no_zero(self.R.z.imag) * epsilon)
        )

        V_inv = Matrix([[a, halftheta], [-halftheta, a]])
        t_tangent = V_inv * self.t
        return [theta, t_tangent[0], t_tangent[1]]

    def storage_D_tangent(self) -> Matrix:
        """
        Note: generated from symforce/notebooks/storage_D_tangent.ipynb
        """
        storage_D_tangent_R = self.R.storage_D_tangent()
        storage_D_tangent_t = self.R.to_rotation_matrix()
        return Matrix.block_matrix(
            [[storage_D_tangent_R, Matrix.zeros(2, 2)], [Matrix.zeros(2, 1), storage_D_tangent_t]]
        )

    def tangent_D_storage(self) -> Matrix:
        """
        Note: generated from symforce/notebooks/tangent_D_storage.ipynb
        """
        tangent_D_storage_R = self.R.tangent_D_storage()
        tangent_D_storage_t = self.R.to_rotation_matrix().T
        return Matrix.block_matrix(
            [[tangent_D_storage_R, Matrix.zeros(1, 2)], [Matrix.zeros(2, 2), tangent_D_storage_t]]
        )

    def retract(self, vec: T.Sequence[T.Scalar], epsilon: T.Scalar = sf.epsilon()) -> Pose2_SE2:
        return LieGroup.retract(self, vec, epsilon)

    def local_coordinates(self, b: Pose2_SE2, epsilon: T.Scalar = sf.epsilon()) -> T.List[T.Scalar]:
        return LieGroup.local_coordinates(self, b, epsilon)

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    @classmethod
    def hat(cls, vec: T.List[T.Scalar]) -> Matrix33:
        R_tangent = [vec[0]]
        t_tangent = [vec[1], vec[2]]
        top_left = Rot2.hat(R_tangent)
        top_right = Matrix21(t_tangent)
        bottom = Matrix13.zero()
        return T.cast(Matrix33, top_left.row_join(top_right).col_join(bottom))
