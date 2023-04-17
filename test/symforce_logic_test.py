# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce.symbolic as sf
from symforce import typing as T
from symforce.test_util import TestCase


class SymforceLogicTest(TestCase):
    """
    Test logic methods
    """

    def test_is_positive(self) -> None:
        """
        Tests:
            sf.is_positive
        """
        self.assertEqual(sf.is_positive(0.1).evalf(), 1.0)
        self.assertEqual(sf.is_positive(0.0).evalf(), 0.0)
        self.assertEqual(sf.is_positive(-0.1).evalf(), 0.0)

    def test_is_negative(self) -> None:
        """
        Tests:
            sf.is_negative
        """
        self.assertEqual(sf.is_negative(0.1).evalf(), 0.0)
        self.assertEqual(sf.is_negative(0.0).evalf(), 0.0)
        self.assertEqual(sf.is_negative(-0.1).evalf(), 1.0)

    def test_is_nonnegative(self) -> None:
        """
        Tests:
            sf.is_nonnegative
        """
        self.assertEqual(sf.is_nonnegative(0.1).evalf(), 1.0)
        self.assertEqual(sf.is_nonnegative(0.0).evalf(), 1.0)
        self.assertEqual(sf.is_nonnegative(-0.1).evalf(), 0.0)

    def test_is_nonpositive(self) -> None:
        """
        Tests:
            sf.is_nonpositive
        """
        self.assertEqual(sf.is_nonpositive(0.1).evalf(), 0.0)
        self.assertEqual(sf.is_nonpositive(0.0).evalf(), 1.0)
        self.assertEqual(sf.is_nonpositive(-0.1).evalf(), 1.0)

    def test_less_equal(self) -> None:
        """
        Tests:
            sf.less_equal
        """
        self.assertEqual(sf.less_equal(0.0, 1.0).evalf(), 1.0)
        self.assertEqual(sf.less_equal(0.0, 0.0).evalf(), 1.0)
        self.assertEqual(sf.less_equal(1.0, 0.0).evalf(), 0.0)

    def test_greater_equal(self) -> None:
        """
        Tests:
            sf.greater_equal
        """
        self.assertEqual(sf.greater_equal(0.0, 1.0).evalf(), 0.0)
        self.assertEqual(sf.greater_equal(0.0, 0.0).evalf(), 1.0)
        self.assertEqual(sf.greater_equal(1.0, 0.0).evalf(), 1.0)

    def test_less(self) -> None:
        """
        Tests:
            sf.less
        """
        self.assertEqual(sf.less(0.0, 1.0).evalf(), 1.0)
        self.assertEqual(sf.less(0.0, 0.0).evalf(), 0.0)
        self.assertEqual(sf.less(1.0, 0.0).evalf(), 0.0)

    def test_greater(self) -> None:
        """
        Tests:
            sf.greater
        """
        self.assertEqual(sf.greater(0.0, 1.0).evalf(), 0.0)
        self.assertEqual(sf.greater(0.0, 0.0).evalf(), 0.0)
        self.assertEqual(sf.greater(1.0, 0.0).evalf(), 1.0)

    SCALARS_TO_CHECK = (-2, -1, -0.9, -0.1, 0.0, 0.1, 0.9, 1, 2)
    UNSAFE_SCALARS_TO_CHECK = (0.0, 1.0)

    @staticmethod
    def sym_bool_to_bool(a: float) -> bool:
        return a > 0

    @staticmethod
    def bool_to_sym_bool(a: bool) -> sf.Scalar:
        return 1 if a else 0

    def check_logical_and(self, scalars: T.Sequence[float], unsafe: bool) -> None:
        for a in scalars:
            for b in scalars:
                with self.subTest(a=a, b=b, unsafe=unsafe):
                    expected = self.bool_to_sym_bool(
                        self.sym_bool_to_bool(a) and self.sym_bool_to_bool(b)
                    )
                    self.assertEqual(float(sf.logical_and(a, b, unsafe=unsafe)), float(expected))

                for c in scalars:
                    with self.subTest(a=a, b=b, c=c, unsafe=unsafe):
                        expected = self.bool_to_sym_bool(
                            self.sym_bool_to_bool(a)
                            and self.sym_bool_to_bool(b)
                            and self.sym_bool_to_bool(c)
                        )
                        self.assertEqual(
                            float(sf.logical_and(a, b, c, unsafe=unsafe)), float(expected)
                        )

    def test_logical_and(self) -> None:
        """
        Tests:
            sf.logical_and
        """
        self.check_logical_and(self.SCALARS_TO_CHECK, False)
        self.check_logical_and(self.UNSAFE_SCALARS_TO_CHECK, True)

    def check_logical_or(self, scalars: T.Sequence[float], unsafe: bool) -> None:
        for a in scalars:
            for b in scalars:
                with self.subTest(a=a, b=b, unsafe=unsafe):
                    expected = self.bool_to_sym_bool(
                        self.sym_bool_to_bool(a) or self.sym_bool_to_bool(b)
                    )
                    self.assertEqual(float(sf.logical_or(a, b, unsafe=unsafe)), float(expected))

                for c in scalars:
                    with self.subTest(a=a, b=b, c=c, unsafe=unsafe):
                        expected = self.bool_to_sym_bool(
                            self.sym_bool_to_bool(a)
                            or self.sym_bool_to_bool(b)
                            or self.sym_bool_to_bool(c)
                        )
                        self.assertEqual(
                            float(sf.logical_or(a, b, c, unsafe=unsafe)), float(expected)
                        )

    def test_logical_or(self) -> None:
        """
        Tests:
            sf.logical_or
        """
        self.check_logical_or(self.SCALARS_TO_CHECK, False)
        self.check_logical_or(self.UNSAFE_SCALARS_TO_CHECK, True)

    def check_logical_not(self, scalars: T.Sequence[float], unsafe: bool) -> None:
        for a in scalars:
            expected = self.bool_to_sym_bool(not self.sym_bool_to_bool(a))
            self.assertEqual(float(sf.logical_not(a, unsafe)), float(expected))

    def test_logical_not(self) -> None:
        """
        Tests:
            sf.logical_not
        """
        self.check_logical_not(self.SCALARS_TO_CHECK, False)
        self.check_logical_not(self.UNSAFE_SCALARS_TO_CHECK, True)


if __name__ == "__main__":
    TestCase.main()
