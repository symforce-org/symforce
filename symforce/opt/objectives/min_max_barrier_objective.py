# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass

import symforce.symbolic as sf
from symforce.opt import barrier_functions
from symforce.opt import noise_models
from symforce.opt import objective
from symforce.opt.residual_block import ResidualBlock


class MinMaxBarrierObjective(objective.TimestepObjective):
    @dataclass
    class Params:
        """
        Fields (same as `barrier_functions.min_max_power_barrier()`):
            x_nominal_lower: x-value of the point at which the error is equal to error_nominal on
                the left-hand side of the barrier function.
            x_nominal_upper: x-value of the point at which the error is equal to error_nominal on
                the right-hand side of the barrier function.
            error_nominal: Error returned when x equals x_nominal_lower or x_nominal_upper.
            dist_zero_to_nominal: The distance from either of the x_nominal points to the region of
                zero error. Must be less than half the distance between x_nominal_lower and
                x_nominal_upper, and must be greater than zero.
        """

        x_nominal_lower: sf.Scalar
        x_nominal_upper: sf.Scalar
        error_nominal: sf.Scalar
        dist_zero_to_nominal: sf.Scalar

    @dataclass
    class ExtraValues:
        """
        Fields:
            unwhitened_residual: The value of each element of the vector after applying the barrier
                function but before applying the cost scaling.
        """

        unwhitened_residual: sf.Matrix

    @staticmethod
    def residual_at_timestep(
        vector: sf.Matrix,
        params: MinMaxBarrierObjective.Params,
        power: sf.Scalar = 1,
        cost_scaling: sf.Scalar = 1,
    ) -> ResidualBlock:
        """
        Returns the residual block for the given timestep, where the residual is computed by
        applying a barrier function to each element of `vector`, and then optionally scaling the
        corresponding cost in the overall optimization problem by `cost_scaling`.

        Args:
            vector: Each element of the given vector has a barrier function and scaling applied to
                it.
            params: Parameters defining the barrier function
            power: Power of the barrier function. Defaults to 1, producing a linear barrier function
                in the residual, which corresponds to a quadratic cost in the overall optimization
                problem.
            cost_scaling: Optional scaling parameter. Corresponds to multiplying the cost in the
                overall optimization problem by a constant.
        """

        # Apply min/max barrier to each element of the vector
        barrier = lambda x: barrier_functions.min_max_power_barrier(
            x=x,
            x_nominal_lower=params.x_nominal_lower,
            x_nominal_upper=params.x_nominal_upper,
            error_nominal=params.error_nominal,
            dist_zero_to_nominal=params.dist_zero_to_nominal,
            power=power,
        )
        unwhitened_residual = vector.applyfunc(barrier)

        noise_model = noise_models.IsotropicNoiseModel(scalar_information=cost_scaling)
        whitened_residual = noise_model.whiten(unwhitened_residual)

        return ResidualBlock(
            residual=whitened_residual,
            extra_values=MinMaxBarrierObjective.ExtraValues(
                unwhitened_residual=unwhitened_residual
            ),
        )
