# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce.symbolic as sf
from symforce import ops
from symforce import typing as T
from symforce.test_util import TestCase
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


class GeoScalarTest(LieGroupOpsTestMixin, TestCase):
    """
    Test a scalar as a geometric class.
    Note the mixin that tests all storage, group and lie group ops.
    """

    @classmethod
    def element(cls) -> T.Scalar:
        return sf.S(3.2)

    def test_construction_by_type(self) -> None:
        """
        Check that we get correect sympy types out from scalar expressions of various forms.
        """
        x, y = sf.symbols("x y")
        for expr in (12, -1.3, sf.S(4), sf.S(12.5), x, x ** 2 + y):
            for cls in (float, sf.Symbol):
                expected = sf.S(expr)
                self.assertEqual(expected, ops.LieGroupOps.from_tangent(cls, [expr]))
                self.assertEqual(expected, ops.StorageOps.from_storage(cls, [expr]))


if __name__ == "__main__":
    TestCase.main()
