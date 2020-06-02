import numpy as np

from symforce import geo
from symforce import sympy as sm
from symforce.test_util import TestCase
from symforce.test_util.group_ops_test_mixin import GroupOpsTestMixin


class GeoComplexTest(GroupOpsTestMixin, TestCase):
    """
    Test the Complex geometric class.
    Note the mixin that tests all storage and group ops.
    """

    @classmethod
    def element(cls):
        # type: () -> geo.Complex
        return geo.Complex(-3.2, 2.8)

    def test_complex_constructors(self):
        # type: () -> None
        """
        Tests:
            Complex.zero
            Complex.symbolic
        """
        zero = geo.Complex.zero()
        self.assertEqual(zero, geo.Complex(0, 0))

        a = geo.Complex.symbolic("a")
        b = geo.Complex.symbolic("b")
        self.assertEqual(a * b, geo.Complex.compose(a, b))

    def complex_operations(self, a, b):
        # type: (geo.Complex, geo.Complex) -> None
        """
        Runs tests on complex operations
        """
        self.assertEqual(a + b, geo.Complex(a.real + b.real, a.imag + b.imag))
        self.assertEqual(
            a * b, geo.Complex(a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real)
        )
        self.assertEqual(-a, geo.Complex(-a.real, -a.imag))
        self.assertEqual(a / 5.0, geo.Complex(a.real / 5.0, a.imag / 5.0))
        d = sm.Symbol("denom")
        self.assertEqual(a / d, geo.Complex(a.real / d, a.imag / d))

    def test_complex_operations_numeric(self):
        # type: () -> None
        """
        Tests (numeric case):
            Complex.__add__
            Complex.__mul__
            Complex.__neg__
            Complex.__truediv__ or Complex.__div__
        """
        a = geo.Complex.random_uniform(-1, 1)
        b = geo.Complex.random_uniform(-1, 1)
        self.complex_operations(a, b)

    def test_complex_operations_symbolic(self):
        # type: () -> None
        """
        Tests (symbolic case):
            Complex.__add__
            Complex.__mul__
            Complex.__neg__
            Complex.__truediv__ or Complex.__div__
        """
        a = geo.Complex.symbolic("a")
        b = geo.Complex.symbolic("b")
        self.complex_operations(a, b)


if __name__ == "__main__":
    TestCase.main()
