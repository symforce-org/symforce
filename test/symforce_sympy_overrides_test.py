# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce.symbolic as sf
from symforce import typing as T
from symforce.test_util import TestCase


class SympyOverridesTest(TestCase):
    """
    Test overrides applied to sympy and symengine
    """

    def test_solve(self) -> None:
        """
        Tests:
            sf.solve
        """

        # Two solutions to (x - 2) * (x + y) == 0
        x, y = sf.symbols("x y")
        solution = sf.solve((x - 2) * (x + y), x)
        self.assertIsInstance(solution, T.List)
        self.assertEqual(set(solution), {2, -y})

        # No solutions to 2 == 0
        solution = sf.solve(2, x)
        self.assertIsInstance(solution, T.List)
        self.assertEqual(set(solution), set())

    def test_derivatives(self) -> None:
        """
        Tests:
            sf.floor derivatives
            sf.sign derivatives
            sf.Mod derivatives
        """
        x, y = sf.symbols("x y")

        self.assertEqual(sf.floor(x).diff(x), 0)
        self.assertEqual(sf.floor(x**2).diff(x), 0)

        self.assertEqual(sf.sign(x).diff(x), 0)
        self.assertEqual(sf.sign(x**2).diff(x), 0)

        def numerical_derivative(
            f: T.Callable[[sf.Scalar], sf.Scalar], x: sf.Scalar, delta: float = 1e-8
        ) -> float:
            return float((f(x + delta) - f(x - delta)) / (2 * delta))

        for nx, ny in ((5, 2), (-5, 2), (5, -2), (-5, -2)):
            self.assertAlmostEqual(
                float(sf.Mod(x, y).diff(x).subs({x: nx, y: ny})),
                numerical_derivative(
                    lambda _x: sf.Mod(x, y).subs(
                        {x: _x, y: ny}  # pylint: disable=cell-var-from-loop
                    ),
                    nx,
                ),
            )

            self.assertAlmostEqual(
                float(sf.Mod(x, y).diff(y).subs({x: nx, y: ny})),
                numerical_derivative(
                    lambda _y: sf.Mod(x, y).subs(
                        {x: nx, y: _y}  # pylint: disable=cell-var-from-loop
                    ),
                    ny,
                ),
            )


if __name__ == "__main__":
    TestCase.main()
