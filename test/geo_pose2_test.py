import numpy as np

from symforce import geo
from symforce.ops import LieGroupOps
from symforce.test_util import TestCase
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


class GeoPose2Test(LieGroupOpsTestMixin, TestCase):
    """
    Test the Pose2 geometric class.
    Note the mixin that tests all storage, group and lie group ops.
    """

    @classmethod
    def element(cls):
        return geo.Pose2.from_tangent([-0.2, 5.3, 1.2])

    def test_lie_exponential(self):
        """
        Tests:
            Pose2.hat
            Pose2.expmap
            Pose2.to_homogenous_matrix
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
        expmap = geo.Pose2.expmap(pertubation, epsilon=self.EPSILON)
        matrix_expected = expmap.to_homogenous_matrix()

        # They should match!
        self.assertNear(hat_exp, matrix_expected, places=5)


if __name__ == "__main__":
    TestCase.main()
