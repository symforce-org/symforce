import logging
import numpy as np
import unittest

from symforce import geo
from symforce import logger
from symforce import sympy as sm
from symforce import types as T
from symforce.ops import LieGroupOps
from symforce.test_util import TestCase
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


class GeoRot3Test(LieGroupOpsTestMixin, TestCase):
    """
    Test the Rot3 geometric class.
    Note the mixin that tests all storage, group and lie group ops.
    """

    @classmethod
    def element(cls) -> geo.Rot3:
        return geo.Rot3.from_axis_angle(geo.V3(1, 0, 0), 1.2)

    def test_default_construct(self) -> None:
        """
        Tests:
            Rot3.__init__
        """
        self.assertEqual(geo.Rot3(), geo.Rot3.identity())

    def test_symbolic_substitution(self) -> None:
        """
        Tests:
            Rot3.subs
        """
        R_1 = geo.Rot3.symbolic("R_1")
        R_2 = geo.Rot3.symbolic("R_2")
        self.assertEqual(R_2, R_1.subs(R_1, R_2))
        self.assertEqual(geo.M(R_2.to_tangent()), geo.M(R_1.to_tangent()).subs(R_1, R_2))
        self.assertEqual(
            geo.Rot3.from_tangent(R_2.to_tangent()),
            geo.Rot3.from_tangent(R_1.to_tangent()).subs(R_1, R_2),
        )

    def test_angle_between(self) -> None:
        """
        Tests:
            Rot3.angle_between
        """
        x_axis = geo.V3(1, 0, 0)
        rot1 = geo.Rot3.from_axis_angle(x_axis, 0.3)
        rot2 = geo.Rot3.from_axis_angle(x_axis, -1.1)
        angle = rot1.angle_between(rot2, epsilon=self.EPSILON)
        self.assertNear(angle, 1.4, places=7)

    def get_rotations_to_test(self) -> T.List[geo.Rot3]:
        """
        Returns a list of rotations to be used in rotation helper method tests.
        """
        rotations_to_test = []
        # Test 90 degree rotations about each principal axis
        for axis_index in range(3):
            for angle in [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0, 2.0 * np.pi]:
                axis = geo.V3.zero()
                axis[axis_index] = 1.0
                rotations_to_test.append(geo.Rot3.from_axis_angle(axis, angle))

        # Test some random rotations
        for _ in range(100):
            rotations_to_test.append(geo.Rot3.random())

        return rotations_to_test

    def test_to_from_rotation_matrix(self) -> None:
        """
        Tests:
            Rot3.from_rotation_matrix
            Rot3.to_rotation_matrix
        """

        # Zero degree rotation
        rot_0 = geo.I33()
        R_0 = geo.Rot3.from_rotation_matrix(rot_0)
        self.assertEqual(R_0, geo.Rot3.identity())
        self.assertEqual(rot_0, R_0.to_rotation_matrix())

        # 180 degree rotation
        rot_180 = geo.Matrix33([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        R_180 = geo.Rot3.from_rotation_matrix(rot_180, epsilon=1e-10)
        self.assertNear(rot_180, R_180.to_rotation_matrix())

        # Check functions are inverses of each other
        for rot in self.get_rotations_to_test():
            rot_transformed = geo.Rot3.from_rotation_matrix(rot.to_rotation_matrix())
            self.assertLieGroupNear(rot_transformed, rot)

    @unittest.expectedFailure
    def test_from_rotation_matrix_theta_equals_180(self) -> None:
        """
        TODO(brad): make this pass
        Tests:
            Rot3.from_rotation_matrix

        Tests that Rot3.from_rotation_matrix creates the correct Rot3 from a
        rotation matrix representing a 180 degree rotation about an axis
        whose components have different signs.
        """

        # 180 degree rotation about axis = [1, -1, 0]
        rot_180_axis = geo.Rot3.from_axis_angle(geo.V3(1, -1, 0).normalized(), sm.pi)
        rot_180_axis_transformed = geo.Rot3.from_rotation_matrix(rot_180_axis.to_rotation_matrix())
        self.assertEqual(
            rot_180_axis.to_rotation_matrix(),
            rot_180_axis_transformed.to_rotation_matrix().simplify(),
        )

    def test_to_from_euler_ypr(self) -> None:
        """
        Tests:
            Rot3.from_euler_ypr
            Rot3.to_euler_ypr
        """

        # Rotations about principal axes
        R_90_yaw = geo.Rot3.from_euler_ypr(np.pi / 2.0, 0, 0)
        self.assertLieGroupNear(R_90_yaw, geo.Rot3.from_axis_angle(geo.V3(0, 0, 1), np.pi / 2.0))
        R_90_pitch = geo.Rot3.from_euler_ypr(0, np.pi / 2.0, 0)
        self.assertLieGroupNear(R_90_pitch, geo.Rot3.from_axis_angle(geo.V3(0, 1, 0), np.pi / 2.0))
        R_90_roll = geo.Rot3.from_euler_ypr(0, 0, np.pi / 2.0)
        self.assertLieGroupNear(R_90_roll, geo.Rot3.from_axis_angle(geo.V3(1, 0, 0), np.pi / 2.0))

        # Check functions are inverses of each other
        for rot in self.get_rotations_to_test():
            rot_transformed = geo.Rot3.from_euler_ypr(*rot.to_euler_ypr(epsilon=1e-14))
            self.assertLieGroupNear(rot_transformed, rot, places=6)

    def test_from_two_unit_vectors(self) -> None:
        """
        Tests:
            Rot3.from_two_unit_vectors
        """
        one = geo.V3(1, 0, 0)
        two = geo.V3(1, 1, 0).normalized()
        rot = geo.Rot3.from_two_unit_vectors(one, two)

        one_rotated = rot * one
        self.assertNear(one_rotated, two)

    def test_random(self) -> None:
        """
        Tests:
            Rot3.random
            Rot3.random_from_uniform_sample
        """
        random_elements = []
        random_from_uniform_samples_elements = []
        for _ in range(100):
            random_element = geo.Rot3.random()
            random_elements.append(random_element)

            u1, u2, u3 = np.random.uniform(low=0.0, high=1.0, size=(3,))
            rand_uniform_sample_element = geo.Rot3.random_from_uniform_samples(u1, u2, u3)
            random_from_uniform_samples_elements.append(rand_uniform_sample_element)

            # Check unit norm
            self.assertNear(random_element.q.squared_norm(), 1.0, places=7)
            self.assertNear(rand_uniform_sample_element.q.squared_norm(), 1.0, places=7)

        for elements in [random_elements, random_from_uniform_samples_elements]:
            # Rotate a point through
            P = geo.V3(0, 0, 1)
            Ps_rotated = [e.evalf() * P for e in elements]

            # Compute angles and check basic stats
            angles = np.array(
                [sm.acos(P.dot(P_rot)[0, 0]) for P_rot in Ps_rotated], dtype=np.float64
            )
            self.assertLess(np.min(angles), 0.3)
            self.assertGreater(np.max(angles), np.pi - 0.3)
            self.assertNear(np.mean(angles), np.pi / 2, places=1)

            # Check that we've included both sides of the double cover
            self.assertLess(min([e.evalf().q.w for e in elements]), -0.8)
            self.assertGreater(max([e.evalf().q.w for e in elements]), 0.8)

            # Plot the sphere to show uniform distribution
            if logger.level == logging.DEBUG and self.verbose:
                from mpl_toolkits.mplot3d import Axes3D
                import matplotlib.pyplot as plt

                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                ax.set_aspect("equal")
                ax.scatter(*zip(point.to_flat_list() for point in Ps_rotated))
                plt.show()

    def test_lie_exponential(self) -> None:
        """
        Tests:
            Rot3.hat
            Rot3.to_tangent
            Rot3.to_rotation_matrix
        """
        element = self.element()
        dim = LieGroupOps.tangent_dim(element)
        pertubation = list(np.random.normal(scale=0.1, size=(dim,)))

        # Compute the hat matrix
        hat = geo.M(element.hat(pertubation))

        # Take the matrix exponential (only supported with sympy)
        import sympy

        hat_exp = geo.M(sympy.expand(sympy.exp(sympy.S(hat.mat))))

        # As a comparison, take the exponential map and convert to a matrix
        expmap = geo.Rot3.from_tangent(pertubation, epsilon=self.EPSILON)
        matrix_expected = expmap.to_rotation_matrix()

        # They should match!
        self.assertNear(hat_exp, matrix_expected, places=5)


if __name__ == "__main__":
    TestCase.main()
