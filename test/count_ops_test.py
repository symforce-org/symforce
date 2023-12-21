# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce.symbolic as sf
from symforce.test_util import TestCase


class CountOpsTest(TestCase):
    """
    Test that symengine.count_ops and sympy.count_ops behave
    as we would expect.
    """

    def test_subtraction(self) -> None:
        """
        Tests that sf.count_ops returns sensible outputs for subtraction and is
        consistent between sympy and symengine.
        """
        x, y, z = sf.symbols("x y z")

        with self.subTest(msg=f"{sf.__name__} counts subtraction as one op"):
            self.assertEqual(1, sf.count_ops(x - y))
            self.assertEqual(1, sf.count_ops(-x + y))
            self.assertEqual(2, sf.count_ops(x - y - z))

        with self.subTest(msg=f"{sf.__name__} counts -x - y as two ops"):
            self.assertEqual(2, sf.count_ops(-x - y))

        with self.subTest(msg=f"{sf.__name__} handles coefficients properly"):
            self.assertEqual(1, sf.count_ops(x + 1 * y))
            self.assertEqual(1, sf.count_ops(x + (-1) * y))
            self.assertEqual(2, sf.count_ops(x + 2.1 * y))
            self.assertEqual(2, sf.count_ops(x + (-2.1) * y))

    def test_division(self) -> None:
        """
        Tests that sf.count_ops returns sensible outputs for division and is
        consistent between sympy and symengine.
        """
        x, y, z = sf.symbols("x y z")

        with self.subTest(msg=f"{sf.__name__} counts division as one op"):
            self.assertEqual(1, sf.count_ops(x / y))
            self.assertEqual(1, sf.count_ops(x ** (-1) * y))
            self.assertEqual(2, sf.count_ops((x / y) / z))

        with self.subTest(msg=f"{sf.__name__} counts x**(-1) * y**(-1) as two ops"):
            self.assertEqual(2, sf.count_ops(x ** (-1) * y ** (-1)))
            self.assertEqual(2, sf.count_ops((1 / x) * (1 / y)))
            self.assertEqual(2, sf.count_ops(1 / (x * y)))

        with self.subTest(msg=f"{sf.__name__} handles exponents properly"):
            self.assertEqual(1, sf.count_ops(x * y**1))
            self.assertEqual(2, sf.count_ops(x * y**2))
            self.assertEqual(2, sf.count_ops(x * y ** (-2)))

    def test_constants(self) -> None:
        """
        Tests that constants are no ops.
        """
        with self.subTest(msg=f"{sf.__name__} counts decimals as 0 ops"):
            self.assertEqual(0, sf.count_ops(1.1))
            self.assertEqual(0, sf.count_ops(-1.1))

        with self.subTest(msg=f"{sf.__name__} counts integers as 0 ops"):
            self.assertEqual(0, sf.count_ops(2))
            self.assertEqual(0, sf.count_ops(-sf.S(2)))

        with self.subTest(msg=f"{sf.__name__} counts rationals as 0 ops"):
            self.assertEqual(0, sf.count_ops(sf.Rational(2, 3)))
            self.assertEqual(0, sf.count_ops(sf.Rational(-2, 3)))


if __name__ == "__main__":
    TestCase.main()
