# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import numpy as np

from symforce import geo
from symforce import sympy as sm
from symforce.test_util import TestCase
from symforce.test_util.group_ops_test_mixin import GroupOpsTestMixin


class GeoDualQuaternionTest(GroupOpsTestMixin, TestCase):
    """
    Test the DualQuaternion geometric class.
    Note the mixin that tests all storage and group ops.
    """

    @classmethod
    def element(cls) -> geo.DualQuaternion:
        return geo.DualQuaternion(
            real_q=geo.Quaternion(xyz=geo.V3(0.1, -0.3, 1.3), w=3.2),
            inf_q=geo.Quaternion(xyz=geo.V3(1.2, 0.3, 0.7), w=0.1),
        )

    def dual_quaternion_operations(self, a: geo.DualQuaternion, b: geo.DualQuaternion) -> None:
        """
        Tests dual quaternion operations
        """
        self.assertEqual(a * b, geo.DualQuaternion.compose(a, b))
        self.assertEqual(a / 5.0, geo.DualQuaternion(a.real_q / 5.0, a.inf_q / 5.0))
        d = sm.Symbol("denom")
        self.assertEqual(a / d, geo.DualQuaternion(a.real_q / d, a.inf_q / d))
        self.assertEqual(a.squared_norm(), a.real_q.squared_norm() + a.inf_q.squared_norm())
        self.assertEqual(a.conj(), geo.DualQuaternion(a.real_q.conj(), a.inf_q.conj()))

    def test_dual_quaternion_operations_numeric(self) -> None:
        """
        Tests (numeric case):
            DualQuaternion.__mul__
            DualQuaternion.__truediv__ or DualQuaternion.__div__
            DualQuaternion.squared_norm
            DualQuaternion.conj
        """
        a_real = geo.Quaternion.unit_random()
        a_inf = geo.Quaternion.unit_random()
        b_real = geo.Quaternion.unit_random()
        b_inf = geo.Quaternion.unit_random()
        a = geo.DualQuaternion(a_real, a_inf)
        b = geo.DualQuaternion(b_real, b_inf)
        self.dual_quaternion_operations(a, b)

    def test_dual_quaternion_operations_symbolic(self) -> None:
        """
        Tests (symbolic case):
            DualQuaternion.__mul__
            DualQuaternion.__truediv__ or DualQuaternion.__div__
            DualQuaternion.squared_norm
            DualQuaternion.conj
        """
        a = geo.DualQuaternion.symbolic("a")
        b = geo.DualQuaternion.symbolic("b")
        self.dual_quaternion_operations(a, b)


if __name__ == "__main__":
    TestCase.main()
