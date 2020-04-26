import numpy as np

from symforce import geo
from symforce.test_util import TestCase
from symforce.test_util.group_ops_test_mixin import GroupOpsTestMixin


class ComplexTest(GroupOpsTestMixin, TestCase):
    """
    Test the Complex geometric class.
    Note the mixin that tests all storage and group ops.
    """

    @classmethod
    def element(cls):
        return geo.Complex(-3.2, 2.8)


if __name__ == "__main__":
    TestCase.main()
