import numpy as np

from symforce import geo
from symforce import sympy as sm
from symforce.ops import LieGroupOps
from symforce.test_util import TestCase
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


class GeoPose2Test(LieGroupOpsTestMixin, TestCase):
    """
    Test the Pose2 geometric class.
    Note the mixin that tests all storage, group and lie group ops.
    """

    @classmethod
    def element(cls) -> geo.Pose2:
        return geo.Pose2.from_tangent([-0.2, 5.3, 1.2])

    def test_lie_exponential(self) -> None:
        """
        Tests:
            Pose2.hat
            Pose2.to_tangent
            Pose2.to_homogenous_matrix
        """
        element = self.element()
        dim = LieGroupOps.tangent_dim(element)
        pertubation = list(float(f) for f in np.random.normal(scale=0.1, size=(dim,)))

        # Compute the hat matrix
        hat = element.hat(pertubation)

        # Take the matrix exponential (only supported with sympy)
        import sympy

        hat_exp = geo.M(sympy.expand(sympy.exp(sympy.S(hat.mat))))

        # As a comparison, take the exponential map and convert to a matrix
        expmap = geo.Pose2.from_tangent(pertubation, epsilon=self.EPSILON)
        matrix_expected = expmap.to_homogenous_matrix()

        # They should match!
        self.assertNear(hat_exp, matrix_expected, places=5)

    def pose2_operations(self, a: geo.Pose2, b: geo.Pose2) -> None:
        """
        Tests Pose2 operations
        """
        self.assertEqual(a * b, a.compose(b))
        self.assertEqual(a * b.t, a.R * b.t + a.t)

    def test_pose2_operations_numeric(self) -> None:
        """
        Tests:
            Pose2.__mul__
        """
        R_a = geo.Rot2.random()
        t_a = geo.V2(np.random.rand(2))
        a = geo.Pose2(R_a, t_a)

        R_b = geo.Rot2.random()
        t_b = geo.V2(np.random.rand(2))
        b = geo.Pose2(R_b, t_b)

        self.pose2_operations(a, b)

    def test_pose2_operations_symbolic(self) -> None:
        """
        Tests:
            Pose2.__mul__
        """
        a = geo.Pose2.symbolic("a")
        b = geo.Pose2.symbolic("b")
        self.pose2_operations(a, b)


if __name__ == "__main__":
    TestCase.main()
