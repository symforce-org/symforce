import numpy as np

from symforce import geo
from symforce import sympy as sm
from symforce.ops import LieGroupOps
from symforce.test_util import TestCase
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


class Pose3Test(LieGroupOpsTestMixin, TestCase):
    """
    Test the Pose3 geometric class.
    Note the mixin that tests all storage, group and lie group ops.
    """

    @classmethod
    def element(cls):
        return geo.Pose3.from_tangent([1.3, 0.2, 1.1, -0.2, 5.3, 1.2])

    def test_lie_exponential(self):
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


if __name__ == "__main__":
    TestCase.main()
