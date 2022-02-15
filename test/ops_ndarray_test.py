# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import unittest
import numpy as np

from symforce import typing as T
from symforce.test_util import TestCase
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


class LieGroupNdarrayTest(LieGroupOpsTestMixin, TestCase):
    """
    Test that ndarrays function as a LieGroup
    Note the mixin that tests all storage, group, and lie group ops.
    """

    @classmethod
    def element(cls) -> np.ndarray:
        return np.array([[1, 2, 3], [4, 5, 6]], dtype=T.Scalar)


if __name__ == "__main__":
    TestCase.main()
