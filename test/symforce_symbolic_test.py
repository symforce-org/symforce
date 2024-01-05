# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce.symbolic as sf
from symforce.test_util import TestCase


class SymforceSymbolicTest(TestCase):
    """
    Test custom methods in symforce.symbolic
    """

    def test_simplify(self) -> None:
        x, y = sf.symbols("x y")
        self.assertEqual(sf.simplify((x + y) ** 2 - x**2 - y**2), 2 * x * y)

    def test_limit(self) -> None:
        x = sf.Symbol("x")
        self.assertEqual(sf.limit(sf.sin(x) / x, x, 0), 1)

    def test_integrate(self) -> None:
        x = sf.Symbol("x")
        self.assertEqual(sf.integrate(2 * x, x), x**2)
        self.assertEqual(sf.integrate(2 * x, (x, 0, 1)), 1)


if __name__ == "__main__":
    TestCase.main()
