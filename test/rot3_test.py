import numpy as np

from symforce import geo
from symforce import sympy as sm
from symforce.ops import LieGroupOps
from symforce.test_util import TestCase
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


class Rot3Test(LieGroupOpsTestMixin, TestCase):
    """
    Test the Rot3 geometric class.
    Note the mixin that tests all storage, group and lie group ops.
    """

    @classmethod
    def element(cls):
        return geo.Rot3.from_axis_angle(geo.V3(1, 0, 0), 1.2)

    def test_default_construct(self):
        """
        Tests:
            Rot3.__init__
        """
        self.assertEqual(geo.Rot3(), geo.Rot3.identity())

    def test_angle_between(self):
        """
        Tests:
            Rot3.angle_between
        """
        x_axis = geo.V3(1, 0, 0)
        rot1 = geo.Rot3.from_axis_angle(x_axis, 0.3)
        rot2 = geo.Rot3.from_axis_angle(x_axis, -1.1)
        angle = rot1.angle_between(rot2, epsilon=self.EPSILON)
        self.assertAlmostEqual(angle, 1.4, places=7)

    def test_from_two_unit_vectors(self):
        """
        Tests:
            Rot3.from_two_unit_vectors
        """
        one = geo.V3(1, 0, 0)
        two = geo.V3(1, 1, 0).normalized()
        rot = geo.Rot3.from_two_unit_vectors(one, two)

        one_rotated = rot * one
        self.assertNear(one_rotated, two)

    def test_lie_exponential(self):
        """
        Tests:
            Rot3.hat
            Rot3.expmap
            Rot3.to_rotation_matrix
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
        expmap = geo.Rot3.expmap(pertubation, epsilon=self.EPSILON)
        matrix_expected = expmap.to_rotation_matrix()

        # They should match!
        self.assertNear(hat_exp, matrix_expected, places=5)


if __name__ == "__main__":
    TestCase.main()
