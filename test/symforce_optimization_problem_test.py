# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from symforce import geo
from symforce import cc_sym
from symforce import typing as T
from symforce.opt.sub_problem import SubProblem
from symforce.opt.residual_block import ResidualBlockWithCustomJacobian
from symforce.opt.optimization_problem import OptimizationProblem
from symforce.python_util import AttrDict
from symforce.test_util import TestCase
from symforce.values import Values


class SymforceOptimizationProblemTest(TestCase):
    """
    Test the OptimizationProblem framework.
    """

    def test_custom_jacobians_throws(self) -> None:
        """
        Create a residual with a custom jacobian, but without the derivative filled out for the
        input, and assert that we raise correctly
        """

        class WrongCustomJacobianSubProblem(SubProblem):
            @dataclass
            class Inputs:
                v: geo.V3
                v0: geo.V3

            inputs: WrongCustomJacobianSubProblem.Inputs

            def build_residuals(self) -> Values:
                residual_blocks = Values()
                residual_blocks["residual"] = ResidualBlockWithCustomJacobian(
                    residual=10 * (self.inputs.v - self.inputs.v0),
                    extra_values=None,
                    custom_jacobians={},
                )
                return residual_blocks

            def optimized_values(self) -> T.List[T.Any]:
                return [self.inputs.v]

        custom_jacobian_subproblem = WrongCustomJacobianSubProblem()
        r = Values(custom_jacobian_subproblem=custom_jacobian_subproblem.build_residuals())
        problem = OptimizationProblem(
            subproblems=AttrDict(custom_jacobian_subproblem=custom_jacobian_subproblem),
            residual_blocks=r,
        )

        with self.assertRaises(ValueError):
            problem.make_numeric_factors("custom_jacobian_problem")

    def test_custom_jacobians(self) -> None:
        """
        Create a residual with a custom jacobian, and check that it's used correctly
        """

        class CustomJacobianSubProblem(SubProblem):
            @dataclass
            class Inputs:
                v: geo.V3
                v0: geo.V3

            inputs: CustomJacobianSubProblem.Inputs

            def build_residuals(self) -> Values:
                residual_blocks = Values()
                residual_blocks["residual"] = ResidualBlockWithCustomJacobian(
                    residual=10 * (self.inputs.v - self.inputs.v0),
                    extra_values=None,
                    custom_jacobians={self.inputs.v: 5 * geo.Matrix.eye(3, 3)},
                )
                return residual_blocks

            def optimized_values(self) -> T.List[T.Any]:
                return [self.inputs.v]

        custom_jacobian_subproblem = CustomJacobianSubProblem()
        r = Values(custom_jacobian_subproblem=custom_jacobian_subproblem.build_residuals())
        problem = OptimizationProblem(
            subproblems=AttrDict(custom_jacobian_subproblem=custom_jacobian_subproblem),
            residual_blocks=r,
        )

        factor = problem.make_numeric_factors("custom_jacobian_problem")[0]
        cc_factor = factor.cc_factor(
            {"custom_jacobian.v": cc_sym.Key("v"), "custom_jacobian.v0": cc_sym.Key("v", 0)},
        )

        cc_values = cc_sym.Values()
        cc_values.set(cc_sym.Key("v"), 2 * np.ones(3))
        cc_values.set(cc_sym.Key("v", 0), np.ones(3))

        linearized_factor = cc_factor.linearized_factor(cc_values)
        self.assertTrue((np.array(linearized_factor.residual.data) == 10 * np.ones(3)).all())
        self.assertTrue(
            (np.array(linearized_factor.jacobian.data).reshape((3, 3)) == 5 * np.eye(3)).all()
        )
        self.assertTrue(
            (np.array(linearized_factor.hessian.data).reshape((3, 3)) == 25 * np.eye(3)).all()
        )
        self.assertTrue((np.array(linearized_factor.rhs.data) == 50 * np.ones(3)).all())


if __name__ == "__main__":
    TestCase.main()
