# mypy: disallow-untyped-defs

import numpy as np

from symforce import geo
from symforce import sympy as sm
from symforce.test_util import TestCase
from symforce.test_util.group_ops_test_mixin import GroupOpsTestMixin


class GeoQuaternionTest(GroupOpsTestMixin, TestCase):
    """
    Test the Quaternion geometric class.
    Note the mixin that tests all storage and group ops.
    """

    @classmethod
    def element(cls):
        # type: () -> geo.Quaternion
        return geo.Quaternion(xyz=geo.V3(0.1, -0.3, 1.3), w=3.2)

    def test_from_rotation_matrix(self):
        # type: () -> None
        """
        Tests:
            Quaternion.from_rotation_matrix
        """

        # Zero degree rotation
        rot0 = np.eye(3)
        q0 = geo.Quaternion.from_rotation_matrix(rot0)
        self.assertNear(q0, geo.Quaternion(xyz=geo.V3(0.0, 0.0, 0.0), w=1.0))

        # Random rotation
        rot = geo.Rot3.random()
        q = geo.Quaternion.from_rotation_matrix(rot.to_rotation_matrix())
        self.assertNear(q, rot.q)

    def test_yaw_angle(self):
        # type: () -> None
        """
        Tests:
            Quaternion.yaw_angle
        """
        q = geo.Quaternion.unit_random()

        # Alternate yaw angle implementation from here: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        my_yaw = sm.atan2_safe(2.0 * (q.x * q.y + q.w * q.z), 1 - 2.0 * (q.y ** 2 + q.z ** 2))
        self.assertNear(q.yaw_angle(), my_yaw)


if __name__ == "__main__":
    TestCase.main()
