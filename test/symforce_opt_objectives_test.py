# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
import numpy as np

from symforce.test_util import TestCase

from symforce import sympy as sm
from symforce import geo
from symforce import ops
from symforce.values import Values
from symforce.opt.objectives import PriorValueObjective


class PriorValueObjectiveTest(TestCase):
    """
    Test the prior value objective. This objective penalizes the L2 distance of a variable at each
    timestep from a desired value.
    """

    def test_vec3_cost(self) -> None:
        # Create symbolic objective
        symbolic_inputs = Values(
            actual=geo.V3.symbolic("actual"),
            desired=geo.V3.symbolic("desired"),
            information_diag=geo.V3.symbolic("cost").to_flat_list(),
            epsilon=sm.Symbol("epsilon"),
            cost_scaling=sm.Symbol("cost_scaling"),
        )
        symbolic_residual_block = PriorValueObjective.residual_at_timestep(
            *symbolic_inputs.values()
        )

        # Create numerical test cases
        zero_residual_inputs = Values(
            actual=geo.V3(1, 2, 3),
            desired=geo.V3(1, 2, 3),
            information_diag=[1, 1, 1],
            epsilon=sm.numeric_epsilon,
            cost_scaling=1,
        )
        residual_block_zero = ops.StorageOps.subs(
            symbolic_residual_block, symbolic_inputs, zero_residual_inputs
        )

        nonzero_residual_inputs = Values(
            actual=geo.V3(1, 2, 3),
            desired=geo.V3(4, 5, 6),
            information_diag=[1, 4, 9],
            epsilon=sm.numeric_epsilon,
            cost_scaling=1,
        )
        residual_block_nonzero = ops.StorageOps.subs(
            symbolic_residual_block, symbolic_inputs, nonzero_residual_inputs
        )

        # Check residual values
        self.assertStorageNear(
            residual_block_zero.residual,
            residual_block_zero.residual.zero(),
        )
        self.assertStorageNear(
            residual_block_nonzero.residual,
            geo.V3(-3, -6, -9),
        )

        # Check the jacobian does not change
        symbolic_jacobian = symbolic_residual_block.residual.jacobian(symbolic_inputs["actual"])
        jacobian_zero_residual = symbolic_jacobian.subs(symbolic_inputs, zero_residual_inputs)
        jacobian_nonzero_residual = symbolic_jacobian.subs(symbolic_inputs, nonzero_residual_inputs)
        self.assertStorageNear(jacobian_zero_residual, geo.M33.eye())
        self.assertStorageNear(jacobian_nonzero_residual, geo.M33.diag([1, 2, 3]))

    def test_rot3_cost(self) -> None:
        # Create symbolic objective
        symbolic_inputs = Values(
            actual=geo.Rot3.symbolic("actual"),
            desired=geo.Rot3.symbolic("desired"),
            information_diag=geo.V3.symbolic("cost").to_flat_list(),
            epsilon=sm.Symbol("epsilon"),
            cost_scaling=sm.Symbol("cost_scaling"),
        )
        symbolic_residual_block = PriorValueObjective.residual_at_timestep(
            *symbolic_inputs.values()
        )

        # Create numerical test cases
        zero_residual_inputs = Values(
            actual=geo.Rot3.from_angle_axis(0.5, geo.V3(0, 0, 1).normalized()),
            desired=geo.Rot3.from_angle_axis(0.5, geo.V3(0, 0, 1).normalized()),
            information_diag=[1, 1, 1],
            epsilon=sm.numeric_epsilon,
            cost_scaling=1,
        )
        residual_block_zero = ops.StorageOps.subs(
            symbolic_residual_block, symbolic_inputs, zero_residual_inputs
        )

        nonzero_residual_inputs = Values(
            actual=geo.Rot3.from_angle_axis(0.5, geo.V3(0, 0, 1).normalized()),
            desired=geo.Rot3.from_angle_axis(0.6, geo.V3(0, 0, 1).normalized()),
            information_diag=[1, 4, 9],
            epsilon=sm.numeric_epsilon,
            cost_scaling=1,
        )
        residual_block_nonzero = ops.StorageOps.subs(
            symbolic_residual_block, symbolic_inputs, nonzero_residual_inputs
        )

        # Check residual values
        self.assertStorageNear(
            residual_block_zero.residual,
            residual_block_zero.residual.zero(),
        )
        self.assertStorageNear(
            residual_block_nonzero.residual,
            geo.V3(0.0, 0.0, -0.3),
        )

        # Check the jacobian does not change
        symbolic_jacobian = symbolic_residual_block.residual.jacobian(symbolic_inputs["actual"])
        jacobian_zero_residual = symbolic_jacobian.subs(symbolic_inputs, zero_residual_inputs)
        self.assertStorageNear(jacobian_zero_residual, geo.M33.eye())

        act = symbolic_inputs["actual"]
        des = symbolic_inputs["desired"]
        unwhited_residual = geo.V3(des.local_coordinates(act))
        cost_scaling = symbolic_inputs["cost_scaling"]
        cost_mat = geo.M33.diag(
            [sm.sqrt(cost_scaling * v) for v in symbolic_inputs["information_diag"]]
        )
        expected_jacobian_nonzero_residual = (cost_mat * unwhited_residual).jacobian(act)

        jacobian_nonzero_residual = symbolic_jacobian.subs(symbolic_inputs, nonzero_residual_inputs)
        expected_jacobian_nonzero_residual = expected_jacobian_nonzero_residual.subs(
            symbolic_inputs, nonzero_residual_inputs
        )
        self.assertStorageNear(
            jacobian_nonzero_residual,
            expected_jacobian_nonzero_residual,
        )

    def test_epsilon_handling(self) -> None:
        inputs = Values(
            actual=geo.Rot3.from_angle_axis(-sm.pi, geo.V3(0, 0, 1).normalized()),
            desired=geo.Rot3.from_angle_axis(sm.pi, geo.V3(0, 0, 1).normalized()),
            information_diag=[1, 1, 1],
            epsilon=sm.numeric_epsilon,
            cost_scaling=1,
        )
        residual_block = PriorValueObjective.residual_at_timestep(*inputs.values())

        self.assertTrue(np.sum(np.isnan(residual_block.residual.to_numpy())) == 0)


if __name__ == "__main__":
    TestCase.main()
