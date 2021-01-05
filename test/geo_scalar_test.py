from symforce import sympy as sm
from symforce import ops
from symforce import types as T
from symforce.test_util import TestCase
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


class GeoScalarTest(LieGroupOpsTestMixin, TestCase):
    """
    Test a scalar as a geometric class.
    Note the mixin that tests all storage, group and lie group ops.
    """

    @classmethod
    def element(cls) -> T.Scalar:
        return sm.S(3.2)

    def test_construction_by_type(self) -> None:
        """
        Check that we get correect sympy types out from scalar expressions of various forms.
        """
        x, y = sm.symbols("x y")
        for expr in (12, -1.3, sm.S(4), sm.S(12.5), x, x ** 2 + y):
            for cls in (float, sm.Symbol):
                expected = sm.S(expr)
                self.assertEqual(expected, ops.LieGroupOps.from_tangent(cls, [expr]))
                self.assertEqual(expected, ops.StorageOps.from_storage(cls, [expr]))


if __name__ == "__main__":
    TestCase.main()
