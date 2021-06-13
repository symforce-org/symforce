import unittest

from symforce import types as T
from symforce.test_util import TestCase
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


class LieGroupListTest(LieGroupOpsTestMixin, TestCase):
    """
    Test that lists function as a lie group
    Note the mixin that tests all storage, group, and lie group ops.
    """

    @classmethod
    def element(cls) -> T.List:
        return [1, 2]


if __name__ == "__main__":
    TestCase.main()
