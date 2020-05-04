import numpy as np
import os
import random
import sys
import unittest

from symforce import logger
from symforce import sympy as sm
from symforce.ops import StorageOps


class TestCase(unittest.TestCase):
    """
    Base class for symforce tests. Adds some useful helpers.
    """

    # Set by the --update flag to tell tests that compare against some saved
    # data to update that data instead of failing
    UPDATE = False

    @staticmethod
    def main():
        """
        Call this to run all tests in scope.
        """
        # Sneak through options to expose to tests
        if "--update" in sys.argv:
            TestCase.UPDATE = True
            sys.argv.remove("--update")

        np.random.seed(42)
        random.seed(42)
        unittest.main()

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

    def compare_or_update(self, path, data):
        """
        Compare the given data to what is saved in path, OR update the saved data if
        the --update flag was passed to the test.

        Args:
            path (str): Path to test data file
            data (str): Data resulting from the test
        """
        if TestCase.UPDATE:
            logger.debug('Updating data at: "{}"'.format(path))

            dirname = os.path.dirname(path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            with open(path, "w") as f:
                f.write(data)
        else:
            logger.debug('Comparing data at: "{}"'.format(path))
            with open(path) as f:
                expected_data = f.read()

            self.assertMultiLineEqual(
                data, expected_data, "Failed, use --update to check diff and commit if desired."
            )
