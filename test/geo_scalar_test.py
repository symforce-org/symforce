# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
import numpy as np

import symforce.symbolic as sf
from symforce import ops
from symforce.test_util import TestCase
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


class GeoScalarTest(LieGroupOpsTestMixin, TestCase):
    """
    Test a scalar as a geometric class.
    Note the mixin that tests all storage, group and lie group ops.
    """

    @classmethod
    def element(cls) -> sf.Scalar:
        return sf.S(3.2)

    def test_construction_by_type(self) -> None:
        """
        Check that we get correct types out from scalar expressions of various forms.
        """
        x, y = sf.symbols("x y")
        scalars = (12, -1.3, sf.S(4), sf.S(12.5), np.double(5.5), x, x**2 + y)
        for scalar in scalars:
            scalar_type = type(scalar)
            from_storage = ops.StorageOps.from_storage(scalar_type, [scalar])
            from_tangent = ops.LieGroupOps.from_tangent(scalar_type, [scalar])
            self.assertEqual(scalar, from_storage)
            self.assertEqual(scalar, from_tangent)
            # Not using `assertIsInstance` here because we expect strict type equality
            self.assertEqual(scalar_type, type(from_storage))
            self.assertEqual(scalar_type, type(from_tangent))

            for new_scalar_type in [type(s) for s in scalars]:
                from_storage_new_type = ops.StorageOps.from_storage(new_scalar_type, [scalar])
                if issubclass(new_scalar_type, sf.Expr) or isinstance(scalar, sf.Expr):
                    # One of the from_storage args is a symbolic type, so we should get a symbolic
                    # type back
                    self.assertIsInstance(from_storage_new_type, sf.Expr)
                else:
                    # We are converting between numeric types, so we should get a numeric type back
                    self.assertEqual(new_scalar_type, type(from_storage_new_type))


if __name__ == "__main__":
    TestCase.main()
