# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import symforce.internal.symbolic as sf
from symforce import typing as T
from symforce.ops.interfaces.lie_group import LieGroup

from .matrix import Matrix
from .matrix import Matrix14
from .matrix import Matrix31
from .matrix import Matrix44
from .matrix import Vector3
from .pose3 import Pose3
from .rot3 import Rot3


class Pose3_SE3(Pose3):
    """
    Group of three-dimensional rigid body transformations - SE(3).

    There is no generated runtime analogue of this class in the `sym` package, which means you
    cannot use it as an input or output of generated functions or as a variable in an optimized
    Values.  This is intentional - in general, you should use the Pose3 class instead of this one,
    because the generated expressions will be significantly more efficient.  If you are sure that
    you need the different behavior of this class, it's here for reference or for use in symbolic
    expressions.

    The storage is a quaternion (x, y, z, w) for rotation followed by position (x, y, z).

    The tangent space is 3 elements for rotation followed by 3 elements for translation in the
    rotated frame, meaning we interpolate the translation in the tangent of the rotating frame for
    lie operations. This can be useful but is more expensive than SO(3) x R3 for often no benefit.
    """

    # -------------------------------------------------------------------------
    # Lie group implementation
    # -------------------------------------------------------------------------

    @classmethod
    def from_tangent(cls, v: T.Sequence[T.Scalar], epsilon: T.Scalar = sf.epsilon()) -> Pose3_SE3:
        R_tangent = (v[0], v[1], v[2])
        t_tangent_vector = Vector3(v[3], v[4], v[5])

        R = Rot3.from_tangent(R_tangent, epsilon=epsilon)
        R_hat = Rot3.hat(R_tangent)
        R_hat_sq = R_hat * R_hat
        R_tangent_vector = Vector3(R_tangent)
        theta = sf.sqrt(R_tangent_vector.squared_norm() + epsilon ** 2)

        V = (
            Matrix.eye(3)
            + (1 - sf.cos(theta)) / (theta ** 2) * R_hat
            + (theta - sf.sin(theta)) / (theta ** 3) * R_hat_sq
        )

        return cls(R, V * t_tangent_vector)

    def to_tangent(self, epsilon: T.Scalar = sf.epsilon()) -> T.List[T.Scalar]:
        R_tangent = self.R.to_tangent(epsilon=epsilon)
        R_tangent_vector = Vector3(R_tangent)
        theta = sf.sqrt(R_tangent_vector.squared_norm() + epsilon)
        R_hat = Rot3.hat(R_tangent)

        half_theta = 0.5 * theta

        V_inv = (
            Matrix.eye(3)
            - 0.5 * R_hat
            + (1 - theta * sf.cos(half_theta) / (2 * sf.sin(half_theta)))
            / (theta * theta)
            * (R_hat * R_hat)
        )
        t_tangent = V_inv * self.t
        return R_tangent_vector.col_join(t_tangent).to_flat_list()

    def storage_D_tangent(self) -> Matrix:
        """
        Note: generated from symforce/notebooks/storage_D_tangent.ipynb
        """
        storage_D_tangent_R = self.R.storage_D_tangent()
        storage_D_tangent_t = self.R.to_rotation_matrix()
        return Matrix.block_matrix(
            [[storage_D_tangent_R, Matrix.zeros(4, 3)], [Matrix.zeros(3, 3), storage_D_tangent_t]]
        )

    def tangent_D_storage(self) -> Matrix:
        """
        Note: generated from symforce/notebooks/tangent_D_storage.ipynb
        """
        tangent_D_storage_R = self.R.tangent_D_storage()
        tangent_D_storage_t = self.R.to_rotation_matrix().T
        return Matrix.block_matrix(
            [[tangent_D_storage_R, Matrix.zeros(3, 3)], [Matrix.zeros(3, 4), tangent_D_storage_t]]
        )

    def retract(self, vec: T.Sequence[T.Scalar], epsilon: T.Scalar = sf.epsilon()) -> Pose3_SE3:
        return LieGroup.retract(self, vec, epsilon)

    def local_coordinates(self, b: Pose3_SE3, epsilon: T.Scalar = sf.epsilon()) -> T.List[T.Scalar]:
        return LieGroup.local_coordinates(self, b, epsilon)

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    @classmethod
    def hat(cls, vec: T.List) -> Matrix44:
        R_tangent = [vec[0], vec[1], vec[2]]
        t_tangent = [vec[3], vec[4], vec[5]]
        top_left = Rot3.hat(R_tangent)
        top_right = Matrix31(t_tangent)
        bottom = Matrix14.zero()
        return T.cast(Matrix44, top_left.row_join(top_right).col_join(bottom))
