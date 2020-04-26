import numpy as np
import random
import sys
import unittest

from symforce import sympy as sm
from symforce.ops import StorageOps


class TestCase(unittest.TestCase):
    """
    Base class for symforce tests. Adds some useful helpers.
    """

    def setUp(self):
        # Store verbosity flag so tests can use
        self.verbose = ("-v" in sys.argv) or ("--verbose" in sys.argv)

    def assertNear(self, actual, desired, places=7, msg="", verbose=True):
        """
        Check that two elements are close. Handles sequences, scalars, and geometry types
        using StorageOps.
        """
        return np.testing.assert_almost_equal(
            actual=StorageOps.evalf(StorageOps.to_storage(actual)),
            desired=StorageOps.evalf(StorageOps.to_storage(desired)),
            decimal=places,
            err_msg=msg,
            verbose=verbose,
        )

    @staticmethod
    def main():
        """
        Call this to run all tests in scope.
        """
        np.random.seed(42)
        random.seed(42)
        unittest.main()
