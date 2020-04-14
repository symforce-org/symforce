import numpy as np
import random
import sys
import unittest

from symforce import sympy as sm


class TestCase(unittest.TestCase):
    """
    Base class for symforce tests. Adds some useful helpers.
    """

    def setUp(self):
        # Store verbosity flag so tests can use
        self.verbose = ('-v' in sys.argv) or ('--verbose' in sys.argv)

    @staticmethod
    def main():
        """
        Call this to run all tests in scope.
        """
        np.random.seed(42)
        random.seed(42)
        unittest.main()
