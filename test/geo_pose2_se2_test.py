# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import numpy as np

import symforce.symbolic as sf
from symforce.ops import LieGroupOps
from symforce.test_util import TestCase
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


class GeoPose2SE2Test(LieGroupOpsTestMixin, TestCase):
    """
    Test the Pose2_SE2 geometric class.
    Note the mixin that tests all storage, group and lie group ops.
    """

    @classmethod
    def element(cls) -> sf.unsupported.Pose2_SE2:
        return sf.unsupported.Pose2_SE2.from_tangent([5.3, 1.2, -0.2])

    def test_lie_exponential(self) -> None:
        """
        Tests:
            Pose2_SE2.hat
            Pose2_SE2.to_tangent
            Pose2_SE2.to_homogenous_matrix
        """
        element = self.element()
        dim = LieGroupOps.tangent_dim(element)
        perturbation = list(float(f) for f in np.random.normal(scale=0.1, size=(dim,)))

        # Compute the hat matrix
        hat = element.hat(perturbation)

        # Take the matrix exponential (only supported with sympy)
        import sympy

        hat_exp = sf.M(sympy.expand(sympy.exp(sympy.S(hat.mat))))

        # As a comparison, take the exponential map and convert to a matrix
        expmap = sf.unsupported.Pose2_SE2.from_tangent(perturbation, epsilon=self.EPSILON)
        matrix_expected = expmap.to_homogenous_matrix()

        # They should match!
        self.assertStorageNear(hat_exp, matrix_expected, places=5)

    def pose2_operations(self, a: sf.unsupported.Pose2_SE2, b: sf.unsupported.Pose2_SE2) -> None:
        """
        Tests Pose2_SE2 operations
        """
        self.assertEqual(a * b, a.compose(b))
        self.assertEqual(a * b.t, a.R * b.t + a.t)

    def test_pose2_operations_numeric(self) -> None:
        """
        Tests:
            Pose2_SE2.__mul__
        """
        R_a = sf.Rot2.random()
        t_a = sf.V2(np.random.rand(2))
        a = sf.unsupported.Pose2_SE2(R_a, t_a)

        R_b = sf.Rot2.random()
        t_b = sf.V2(np.random.rand(2))
        b = sf.unsupported.Pose2_SE2(R_b, t_b)

        self.pose2_operations(a, b)

    def test_pose2_operations_symbolic(self) -> None:
        """
        Tests:
            Pose2_SE2.__mul__
        """
        a = sf.unsupported.Pose2_SE2.symbolic("a")
        b = sf.unsupported.Pose2_SE2.symbolic("b")
        self.pose2_operations(a, b)


if __name__ == "__main__":
    TestCase.main()
