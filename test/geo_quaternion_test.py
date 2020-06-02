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


if __name__ == "__main__":
    TestCase.main()
