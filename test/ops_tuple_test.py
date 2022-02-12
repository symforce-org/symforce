# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import unittest

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
        return (1, 2)

    @unittest.expectedFailure
    def test_storage_ops(self) -> None:
        LieGroupOpsTestMixin.test_storage_ops(self)

    @unittest.expectedFailure
    def test_lie_group_ops(self) -> None:
        LieGroupOpsTestMixin.test_lie_group_ops(self)

    @unittest.expectedFailure
    def test_manifold_ops_match_group_ops_definitions(self) -> None:
        LieGroupOpsTestMixin.test_manifold_ops_match_group_ops_definitions(self)

    @unittest.expectedFailure
    def test_storage_D_tangent(self) -> None:
        LieGroupOpsTestMixin.test_storage_D_tangent(self)

    def test_tangent_D_storage(self) -> None:
        LieGroupOpsTestMixin.test_tangent_D_storage(self)


if __name__ == "__main__":
    TestCase.main()
