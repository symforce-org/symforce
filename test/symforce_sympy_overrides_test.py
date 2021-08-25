from symforce import sympy as sm
from symforce import types as T
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


if __name__ == "__main__":
    TestCase.main()
