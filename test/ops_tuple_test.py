# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from symforce import typing as T
from symforce.test_util import TestCase
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


class LieGroupTupleTest(LieGroupOpsTestMixin, TestCase):
    """
    Test that tuples function as a LieGroup
    Note the mixin that tests all storage, group, and lie group ops.
    """

    @classmethod
    def element(cls) -> T.Tuple:
        return (1.0, 2.0)


if __name__ == "__main__":
    TestCase.main()
