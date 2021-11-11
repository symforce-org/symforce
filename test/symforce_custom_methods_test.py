import unittest
import numpy as np

from symforce import geo
from symforce import typing as T
from symforce import sympy as sm
from symforce.test_util import TestCase


class SymforceCustomMethodsTest(TestCase):
    """
    Test the custom methods added by add_custom_methods in initialization.py
    """

    def test_arg_maxes(self) -> None:
        """
        Tests:
            symforce.sympy.argmax_onehot
            symforce.sympy.argmax
        Check that the argmax functions return the correct output
        """

        vals = [-100, 34, 34.1, 10]
        self.assertEqual([0, 0, 1, 0], sm.argmax_onehot(vals))
        self.assertEqual(2, sm.argmax(vals))

        vals = [0]
        self.assertEqual([1], sm.argmax_onehot(vals))
        self.assertEqual(0, sm.argmax(vals))

        vals = [1.0, 1.0, 1.0]
        self.assertEqual([1, 0, 0], sm.argmax_onehot(vals))
        self.assertEqual(0, sm.argmax(vals))

        vals = [-10, 3.0, 2.0, 3.0]
        self.assertEqual([0, 1, 0, 0], sm.argmax_onehot(vals))
        self.assertEqual(1, sm.argmax(vals))

    def test_arg_maxes_other_sequences(self) -> None:
        """
        Tests:
            symforce.sympy.argmax_onehot
            symforce.sympy.argmax
        Check that the argmax functions work on non-list sequences
        """
        vals_range = range(5)
        self.assertEqual([0, 0, 0, 0, 1], sm.argmax_onehot(vals_range))
        self.assertEqual(4, sm.argmax(vals_range))

        vals_v3 = geo.V3(3.3, 3.31, -3.31)
        self.assertEqual([0, 1, 0], sm.argmax_onehot(vals_v3))
        self.assertEqual(1, sm.argmax(vals_v3))

        vals_arr = np.array([30, -32, 23, 23], dtype=T.Scalar)
        self.assertEqual([1, 0, 0, 0], sm.argmax_onehot(vals_arr))
        self.assertEqual(0, sm.argmax(vals_arr))


if __name__ == "__main__":
    TestCase.main()
