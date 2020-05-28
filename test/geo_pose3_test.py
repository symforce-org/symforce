# mypy: disallow-untyped-defs

import numpy as np

from symforce import geo
from symforce import sympy as sm
from symforce import types as T
from symforce.ops import LieGroupOps
from symforce.test_util import TestCase
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


class GeoPose3Test(LieGroupOpsTestMixin, TestCase):
    """
    Test the Pose3 geometric class.
    Note the mixin that tests all storage, group and lie group ops.
    """

    @classmethod
    def element(cls):
        # type: () -> geo.Pose3
        return geo.Pose3.from_tangent([1.3, 0.2, 1.1, -0.2, 5.3, 1.2])

    def test_lie_exponential(self):
        # type: () -> None
        """
        Tests:
            Pose3.hat
            Pose3.expmap
            Pose3.to_homogenous_matrix
        """
        element = self.element()
        dim = LieGroupOps.tangent_dim(element)
        pertubation = list(np.random.normal(scale=0.1, size=(dim,)))

        # Compute the hat matrix
        hat = geo.M(element.hat(pertubation))

        # Take the matrix exponential (only supported with sympy)
        import sympy

        hat_exp = geo.M(sympy.expand(sympy.exp(sympy.Matrix(hat))))

        # As a comparison, take the exponential map and convert to a matrix
        expmap = geo.Pose3.expmap(pertubation, epsilon=self.EPSILON)
        matrix_expected = expmap.to_homogenous_matrix()

        # They should match!
        self.assertNear(hat_exp, matrix_expected, places=5)

    def pose3_operations(self, a, b):
        # type: (geo.Pose3, geo.Pose3) -> None
        """
        Tests Pose3 operations
        """
        self.assertEqual(a * b, a.compose(b))
        self.assertEqual(a * b.t, a.R * b.t + a.t)

    def test_pose3_operations_numeric(self):
        # type: () -> None
        """
        Tests (numeric):
            Pose3.__mul__
        """
        R_a = geo.Rot3.random()
        t_a = geo.V3(np.random.rand(3))
        a = geo.Pose3(R_a, t_a)

        R_b = geo.Rot3.random()
        t_b = geo.V3(np.random.rand(3))
        b = geo.Pose3(R_b, t_b)

        self.pose3_operations(a, b)

    def test_pose3_operations_symbolic(self):
        # type: () -> None
        """
        Tests (symbolic):
            Pose3.__mul__
        """
        a = geo.Pose3.symbolic("a")
        b = geo.Pose3.symbolic("b")
        self.pose3_operations(a, b)


if __name__ == "__main__":
    TestCase.main()
