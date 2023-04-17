# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import numpy as np

import symforce.symbolic as sf
from symforce.test_util import TestCase


class SymforceCustomMethodsTest(TestCase):
    """
    Test the custom methods added by in the "custom functions" section in symbolic.py
    """

    def test_arg_maxes(self) -> None:
        """
        Tests:
            sf.argmax_onehot
            sf.argmax
        Check that the argmax functions return the correct output
        """

        vals = [-100, 34, 34.1, 10]
        self.assertEqual([0, 0, 1, 0], sf.argmax_onehot(vals))
        self.assertEqual(2, sf.argmax(vals))

        vals = [0]
        self.assertEqual([1], sf.argmax_onehot(vals))
        self.assertEqual(0, sf.argmax(vals))

        vals = [1.0, 1.0, 1.0]
        self.assertEqual([1, 0, 0], sf.argmax_onehot(vals))
        self.assertEqual(0, sf.argmax(vals))

        vals = [-10, 3.0, 2.0, 3.0]
        self.assertEqual([0, 1, 0, 0], sf.argmax_onehot(vals))
        self.assertEqual(1, sf.argmax(vals))

    def test_arg_maxes_other_sequences(self) -> None:
        """
        Tests:
            sf.argmax_onehot
            sf.argmax
        Check that the argmax functions work on non-list sequences
        """
        vals_range = range(5)
        self.assertEqual([0, 0, 0, 0, 1], sf.argmax_onehot(vals_range))
        self.assertEqual(4, sf.argmax(vals_range))

        vals_v3 = sf.V3(3.3, 3.31, -3.31)
        self.assertEqual([0, 1, 0], sf.argmax_onehot(vals_v3))
        self.assertEqual(1, sf.argmax(vals_v3))

        vals_arr = np.array([30, -32, 23, 23], dtype=sf.Scalar)
        self.assertEqual([1, 0, 0, 0], sf.argmax_onehot(vals_arr))
        self.assertEqual(0, sf.argmax(vals_arr))

    def test_clamp(self) -> None:
        """
        Tests:
            sf.clamp
        """
        self.assertEqual(1, sf.clamp(-10, 1, 5))
        self.assertEqual(3, sf.clamp(3, 1, 5))
        self.assertEqual(5, sf.clamp(10, 1, 5))


if __name__ == "__main__":
    TestCase.main()
