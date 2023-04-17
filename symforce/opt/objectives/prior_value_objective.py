# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass

import symforce.symbolic as sf
from symforce import typing as T
from symforce.ops import LieGroupOps
from symforce.opt import noise_models
from symforce.opt import objective
from symforce.opt.residual_block import ResidualBlock


class PriorValueObjective(objective.TimestepObjective):
    @dataclass
    class ExtraValues:
        """
        Fields:
            unwhitened_residual: The tangent-space error between `actual` and `desired`.
        """

        unwhitened_residual: sf.Matrix

    @staticmethod
    def residual_at_timestep(
        actual: T.Element,
        desired: T.Element,
        information_diag: T.Sequence[sf.Scalar],
        epsilon: sf.Scalar,
        cost_scaling: sf.Scalar = 1,
    ) -> ResidualBlock:
        """
        Returns the residual block for the given timestep, where the residual is computed as the
        weighted tangent-space difference between `actual` and `desired`.

        Args:
            actual: Typically an element to be optimized in the optimization problem.
            desired: The desired value of the element to be optimized, or equivalently the prior we
                have on `actual`.
            information_diag: List of scalars defining how each element of the tangent-space vector
                representing the transform between `actual` and `desired` is weighted. For example,
                if `actual` and `desired` are 3D positions represented by sf.V3 objects,
                the first, second, and third elements of `information_diag` represent how the x, y,
                and z errors are weighted.
            epsilon: A small number used to avoid singularities.
            cost_scaling: Optional scaling parameter. Corresponds to multiplying the cost in the
                overall optimization problem by a constant.
        """
        unwhitened_residual = sf.Matrix(LieGroupOps.local_coordinates(desired, actual, epsilon))
        noise_model = noise_models.DiagonalNoiseModel(
            information_diag=[cost_scaling * c for c in information_diag]
        )
        whitened_residual = noise_model.whiten(unwhitened_residual)

        return ResidualBlock(
            residual=whitened_residual,
            extra_values=PriorValueObjective.ExtraValues(unwhitened_residual=unwhitened_residual),
        )
