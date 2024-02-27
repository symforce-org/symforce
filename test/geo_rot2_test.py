# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import logging

import numpy as np
import numpy.typing as npt

import symforce.symbolic as sf
from symforce import logger
from symforce.ops import LieGroupOps
from symforce.test_util import TestCase
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


class GeoRot2Test(LieGroupOpsTestMixin, TestCase):
    """
    Test the Rot2 geometric class.
    Note the mixin that tests all storage, group and lie group ops.
    """

    @classmethod
    def element(cls) -> sf.Rot2:
        return sf.Rot2.from_tangent([1.3])

    def test_default_construct(self) -> None:
        """
        Tests:
            Rot2.__init__
        """
        self.assertEqual(sf.Rot2(), sf.Rot2.identity())

    def test_symbolic_constructor(self) -> None:
        """
        Tests:
            Rot2.symbolic
        """
        rot = sf.Rot2.symbolic("rot")
        comp = sf.Complex.symbolic("rot")
        self.assertEqual(rot, sf.Rot2(comp))

    def test_angle_constructor(self) -> None:
        """
        Tests:
            Rot2.from_angle
        """
        rot1 = sf.Rot2.from_angle(1.5)
        rot2 = sf.Rot2.from_tangent([1.5])
        self.assertEqual(rot1, rot2)

    def test_from_to_angle(self) -> None:
        """
        Tests:
            Rot2.from_angle
            Rot2.to_angle
        """
        for angle, angle_gt in zip(
            [0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
            [0.0, np.pi / 2, np.pi, -np.pi / 2, 0.0],
        ):
            rot = sf.Rot2.from_angle(angle).evalf()
            self.assertLess(abs(angle_gt - rot.to_angle()), 1e-8)

    def test_lie_exponential(self) -> None:
        """
        Tests:
            Rot2.hat
            Rot2.to_tangent
            Rot2.to_rotation_matrix
        """
        element = self.element()
        dim = LieGroupOps.tangent_dim(element)
        perturbation = list(np.random.normal(scale=0.1, size=(dim,)))

        # Compute the hat matrix
        hat = element.hat(perturbation)

        # Take the matrix exponential (only supported with sympy)
        import sympy

        hat_exp = sf.M(sympy.expand(sympy.exp(sympy.S(hat.mat))))

        # As a comparison, take the exponential map and convert to a matrix
        expmap = sf.Rot2.from_tangent(perturbation, epsilon=self.EPSILON)
        matrix_expected = expmap.to_rotation_matrix()

        # They should match!
        self.assertStorageNear(hat_exp, matrix_expected, places=5)

    def test_random(self) -> None:
        """
        Tests:
            Rot2.random
            Rot2.random_from_uniform_sample
        """
        random_elements = []
        random_from_uniform_sample_elements = []
        for _ in range(200):
            random_element = sf.Rot2.random()
            random_elements.append(random_element)

            u1 = np.random.uniform(low=0.0, high=1.0)
            rand_uniform_sample_element = sf.Rot2.random_from_uniform_sample(u1)
            random_from_uniform_sample_elements.append(rand_uniform_sample_element)

            # Check unit norm
            self.assertStorageNear(random_element.z.squared_norm(), 1.0, places=7)
            self.assertStorageNear(rand_uniform_sample_element.z.squared_norm(), 1.0, places=7)

        for elements in [random_elements, random_from_uniform_sample_elements]:
            # Rotate a point through
            P = sf.V2(0, 1)
            Ps_rotated = [e.evalf() * P for e in elements]

            # Compute angles and check basic stats
            angles: npt.NDArray[np.float64] = np.array(
                [sf.acos(P.dot(P_rot)) for P_rot in Ps_rotated], dtype=np.float64
            )

            self.assertLess(np.min(angles), 0.3)
            self.assertGreater(np.max(angles), np.pi - 0.3)
            self.assertStorageNear(np.mean(angles), np.pi / 2, places=1)

            # Plot the circle to show uniform distribution
            if logger.level == logging.DEBUG and self.verbose:
                import matplotlib.pyplot as plt

                _, ax = plt.subplots()
                x = [point[0] for point in Ps_rotated]
                y = [point[1] for point in Ps_rotated]
                ax.plot(x, y, ".")
                ax.set_aspect("equal")
                plt.show()


if __name__ == "__main__":
    TestCase.main()
