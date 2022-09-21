# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import symforce

symforce.set_epsilon_to_symbol()

import symforce.symbolic as sf
from symforce import cc_sym
from symforce import typing as T
from symforce.opt.optimization_problem import OptimizationProblem
from symforce.opt.residual_block import ResidualBlock
from symforce.opt.residual_block import ResidualBlockWithCustomJacobian
from symforce.opt.sub_problem import SubProblem
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
                v: sf.V3
                v0: sf.V3

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
                v: sf.V3
                v0: sf.V3
                a: sf.V1

            inputs: CustomJacobianSubProblem.Inputs

            def build_residuals(self) -> Values:
                residual_blocks = Values()
                residual_blocks["residual"] = ResidualBlockWithCustomJacobian(
                    residual=10 * (self.inputs.v - self.inputs.v0),
                    extra_values=None,
                    custom_jacobians={self.inputs.v: 5 * sf.Matrix.eye(3, 3)},
                )
                residual_blocks["residual_a"] = ResidualBlock(
                    residual=self.inputs.a,
                )
                return residual_blocks

            def optimized_values(self) -> T.List[T.Any]:
                return [self.inputs.v, self.inputs.a]

        custom_jacobian_subproblem = CustomJacobianSubProblem()
        r = Values(custom_jacobian_subproblem=custom_jacobian_subproblem.build_residuals())
        problem = OptimizationProblem(
            subproblems=AttrDict(custom_jacobian_subproblem=custom_jacobian_subproblem),
            residual_blocks=r,
        )

        factor = problem.make_numeric_factors("custom_jacobian_problem")[0]
        cc_factor = factor.cc_factor(
            {
                "custom_jacobian.v": cc_sym.Key("v"),
                "custom_jacobian.v0": cc_sym.Key("v", 0),
                "custom_jacobian.a": cc_sym.Key("a"),
            },
        )

        cc_values = cc_sym.Values()
        cc_values.set(cc_sym.Key("v"), 2 * np.ones(3))
        cc_values.set(cc_sym.Key("v", 0), np.ones(3))
        cc_values.set(cc_sym.Key("a"), np.ones(1))

        linearized_factor = cc_factor.linearized_factor(cc_values)
        residual = np.array(linearized_factor.residual.data)
        self.assertTrue((residual[0:3] == 10 * np.ones(3)).all())
        self.assertTrue((residual[3] == 1))

        jacobian = np.array(linearized_factor.jacobian.data).reshape((4, 4))
        self.assertTrue((jacobian[0:3, 0:3] == 5 * np.eye(3)).all())
        self.assertTrue(jacobian[3, 3] == 1)

        hessian = np.array(linearized_factor.hessian.data).reshape((4, 4))
        self.assertTrue((hessian[0:3, 0:3] == 25 * np.eye(3)).all())
        self.assertTrue(hessian[3, 3] == 1)

        rhs = np.array(linearized_factor.rhs.data)
        self.assertTrue((rhs[0:3] == 50 * np.ones(3)).all())
        self.assertTrue(rhs[3] == 1)

    def test_optimized_values(self) -> None:
        """
        Tests that a TypeError is raised when a subproblem has an optimized_values that returns a
        different type from the type used in the subproblem Inputs.
        """

        class CustomSubProblem(SubProblem):
            @dataclass
            class Inputs:
                rot: sf.Rot3
                rot0: sf.Rot3

            inputs: CustomSubProblem.Inputs

            def build_residuals(self) -> Values:
                residual_blocks = Values()
                residual_blocks["residual"] = ResidualBlock(
                    residual=sf.V1(self.inputs.rot.angle_between(self.inputs.rot0, epsilon=0)),
                    extra_values=None,
                )
                return residual_blocks

            def optimized_values(self) -> T.List[T.Any]:
                return [sf.V4(self.inputs.rot.to_storage())]

        custom_subproblem = CustomSubProblem()
        r = Values(custom_jacobian_subproblem=custom_subproblem.build_residuals())
        problem = OptimizationProblem(
            subproblems=AttrDict(custom_subproblem=custom_subproblem),
            residual_blocks=r,
        )

        output_dir = self.make_output_dir("sf_values_codegen_test")

        with self.assertRaises(TypeError):
            problem.generate(name="optimized_values_test", output_dir=output_dir, namespace="sym")


if __name__ == "__main__":
    TestCase.main()
