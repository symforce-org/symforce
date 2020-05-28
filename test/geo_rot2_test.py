# mypy: disallow-untyped-defs

import numpy as np
import logging

from symforce import geo
from symforce import logger
from symforce import sympy as sm
from symforce.ops import LieGroupOps
from symforce.test_util import TestCase
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


class GeoRot2Test(LieGroupOpsTestMixin, TestCase):
    """
    Test the Rot2 geometric class.
    Note the mixin that tests all storage, group and lie group ops.
    """

    @classmethod
    def element(cls):
        # type: () -> geo.Rot2
        return geo.Rot2.from_tangent([1.3])

    def test_default_construct(self):
        # type: () -> None
        """
        Tests:
            Rot2.__init__
        """
        self.assertEqual(geo.Rot2(), geo.Rot2.identity())

    def test_symbolic_constructor(self):
        # type: () -> None
        """
        Tests:
            Rot2.symbolic
        """
        rot = geo.Rot2.symbolic("rot")
        comp = geo.Complex.symbolic("rot")
        self.assertEqual(rot, geo.Rot2(comp))

    def test_lie_exponential(self):
        # type: () -> None
        """
        Tests:
            Rot2.hat
            Rot2.expmap
            Rot2.to_rotation_matrix
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
        expmap = geo.Rot2.expmap(pertubation, epsilon=self.EPSILON)
        matrix_expected = expmap.to_rotation_matrix()

        # They should match!
        self.assertNear(hat_exp, matrix_expected, places=5)

    def test_random(self):
        # type: () -> None
        """
        Tests:
            Rot2.random
            Rot2.random_from_uniform_sample
        """
        random_elements = []
        random_from_uniform_sample_elements = []
        for _ in range(200):
            random_element = geo.Rot2.random()
            random_elements.append(random_element)

            u1 = np.random.uniform(low=0.0, high=1.0)
            rand_uniform_sample_element = geo.Rot2.random_from_uniform_sample(u1)
            random_from_uniform_sample_elements.append(rand_uniform_sample_element)

            # Check unit norm
            self.assertNear(random_element.z.squared_norm(), 1.0, places=7)
            self.assertNear(rand_uniform_sample_element.z.squared_norm(), 1.0, places=7)

        for elements in [random_elements, random_from_uniform_sample_elements]:
            # Rotate a point through
            P = geo.V2(0, 1)
            Ps_rotated = [e.evalf() * P for e in elements]

            # Compute angles and check basic stats
            angles = np.array([sm.acos(P.dot(P_rot)) for P_rot in Ps_rotated], dtype=np.float64)

            self.assertLess(np.min(angles), 0.3)
            self.assertGreater(np.max(angles), np.pi - 0.3)
            self.assertNear(np.mean(angles), np.pi / 2, places=1)

            # Plot the circle to show uniform distribution
            if logger.level == logging.DEBUG and self.verbose:
                import matplotlib.pyplot as plt

                _, ax = plt.subplots()
                x = zip(*Ps_rotated)[0]
                y = zip(*Ps_rotated)[1]
                ax.plot(x, y, ".")
                ax.set_aspect("equal")
                plt.show()


if __name__ == "__main__":
    TestCase.main()
