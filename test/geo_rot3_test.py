import logging
import numpy as np

from symforce import geo
from symforce import logger
from symforce import sympy as sm
from symforce.ops import LieGroupOps
from symforce.test_util import TestCase
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


class GeoRot3Test(LieGroupOpsTestMixin, TestCase):
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

    def test_random(self):
        """
        Tests:
            Rot3.random
        """
        np.random.seed(0)

        elements = []
        for _ in range(100):
            element = geo.Rot3.random()
            elements.append(element)

            # Check unit norm
            self.assertAlmostEqual(element.q.squared_norm(), 1.0, places=7)

            # Check positive w (to go on one side of double cover)
            self.assertGreaterEqual(element.q.w, 0.0)

        # Rotate a point through
        P = geo.V3(0, 0, 1)
        Ps_rotated = [e * P for e in elements]

        # Compute angles and check basic stats
        angles = np.array([sm.acos(P.dot(P_rot)) for P_rot in Ps_rotated], dtype=np.float64)
        self.assertLess(np.min(angles), 0.3)
        self.assertGreater(np.max(angles), np.pi - 0.3)
        self.assertAlmostEqual(np.mean(angles), np.pi / 2, places=1)

        # Plot the sphere to show uniform distribution
        if logger.level == logging.DEBUG and self.verbose:
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.set_aspect("equal")
            ax.scatter(*zip(*Ps_rotated))
            plt.show()

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
