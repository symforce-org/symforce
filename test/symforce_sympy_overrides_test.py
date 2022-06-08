# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from symforce import sympy as sm
from symforce import typing as T
from symforce.test_util import TestCase


class SympyOverridesTest(TestCase):
    """
    Test overrides applied to sympy and symengine
    """

    def test_solve(self) -> None:
        """
        Tests:
            sm.solve
        """

        # Two solutions to (x - 2) * (x + y) == 0
        x, y = sm.symbols("x y")
        solution = sm.solve((x - 2) * (x + y), x)
        self.assertIsInstance(solution, T.List)
        self.assertEqual(set(solution), {2, -y})

        # No solutions to 2 == 0
        solution = sm.solve(2, x)
        self.assertIsInstance(solution, T.List)
        self.assertEqual(set(solution), set())

    def test_derivatives(self) -> None:
        """
        Tests:
            sm.floor derivatives
            sm.sign derivatives
            sm.Mod derivatives
        """
        x, y = sm.symbols("x y")

        self.assertEqual(sm.floor(x).diff(x), 0)
        self.assertEqual(sm.floor(x ** 2).diff(x), 0)

        self.assertEqual(sm.sign(x).diff(x), 0)
        self.assertEqual(sm.sign(x ** 2).diff(x), 0)

        def numerical_derivative(
            f: T.Callable[[T.Scalar], T.Scalar], x: T.Scalar, delta: float = 1e-8
        ) -> float:
            return float((f(x + delta) - f(x - delta)) / (2 * delta))

        for nx, ny in ((5, 2), (-5, 2), (5, -2), (-5, -2)):
            self.assertAlmostEqual(
                float(sm.Mod(x, y).diff(x).subs({x: nx, y: ny})),
                numerical_derivative(lambda _x: sm.Mod(x, y).subs({x: _x, y: ny}), nx),
            )

            self.assertAlmostEqual(
                float(sm.Mod(x, y).diff(y).subs({x: nx, y: ny})),
                numerical_derivative(lambda _y: sm.Mod(x, y).subs({x: nx, y: _y}), ny),
            )


if __name__ == "__main__":
    TestCase.main()
