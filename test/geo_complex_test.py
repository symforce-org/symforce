# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce.symbolic as sf
from symforce.test_util import TestCase
from symforce.test_util.group_ops_test_mixin import GroupOpsTestMixin


class GeoComplexTest(GroupOpsTestMixin, TestCase):
    """
    Test the Complex geometric class.
    Note the mixin that tests all storage and group ops.
    """

    @classmethod
    def element(cls) -> sf.Complex:
        return sf.Complex(-3.2, 2.8)

    def test_complex_constructors(self) -> None:
        """
        Tests:
            Complex.zero
            Complex.symbolic
        """
        zero = sf.Complex.zero()
        self.assertEqual(zero, sf.Complex(0, 0))

        a = sf.Complex.symbolic("a")
        b = sf.Complex.symbolic("b")
        self.assertEqual(a * b, sf.Complex.compose(a, b))

    def complex_operations(self, a: sf.Complex, b: sf.Complex) -> None:
        """
        Runs tests on complex operations
        """
        self.assertEqual(a + b, sf.Complex(a.real + b.real, a.imag + b.imag))
        self.assertEqual(
            a * b, sf.Complex(a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real)
        )
        self.assertEqual(-a, sf.Complex(-a.real, -a.imag))
        self.assertEqual(a / 5.0, sf.Complex(a.real / 5.0, a.imag / 5.0))
        d = sf.Symbol("denom")
        self.assertEqual(a / d, sf.Complex(a.real / d, a.imag / d))

    def test_complex_operations_numeric(self) -> None:
        """
        Tests (numeric case):
            Complex.__add__
            Complex.__mul__
            Complex.__neg__
            Complex.__truediv__ or Complex.__div__
        """
        a = sf.Complex.random_uniform(-1, 1)
        b = sf.Complex.random_uniform(-1, 1)
        self.complex_operations(a, b)

    def test_complex_operations_symbolic(self) -> None:
        """
        Tests (symbolic case):
            Complex.__add__
            Complex.__mul__
            Complex.__neg__
            Complex.__truediv__ or Complex.__div__
        """
        a = sf.Complex.symbolic("a")
        b = sf.Complex.symbolic("b")
        self.complex_operations(a, b)


if __name__ == "__main__":
    TestCase.main()
