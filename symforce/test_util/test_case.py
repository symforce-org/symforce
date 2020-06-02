import numpy as np
import os
import random
import sys
import unittest

from symforce import geo
from symforce import logger
from symforce import python_util
from symforce import sympy as sm
from symforce import types as T
from symforce.ops import StorageOps
from symforce.ops import GroupOps
from symforce.ops import LieGroupOps


class TestCase(unittest.TestCase):
    """
    Base class for symforce tests. Adds some useful helpers.
    """

    # Set by the --update flag to tell tests that compare against some saved
    # data to update that data instead of failing
    UPDATE = False

    @staticmethod
    def main():
        # type: () -> None
        """
        Call this to run all tests in scope.
        """
        # Sneak through options to expose to tests
        if "--update" in sys.argv:
            TestCase.UPDATE = True
            sys.argv.remove("--update")

        unittest.main()

    def setUp(self):
        # type: () -> None
        np.random.seed(42)
        random.seed(42)
        # Store verbosity flag so tests can use
        self.verbose = ("-v" in sys.argv) or ("--verbose" in sys.argv)

    def assertNear(
        self,
        actual,  # type: T.Any
        desired,  # type: T.Any
        places=7,  # type: int
        msg="",  # type: str
        verbose=True,  # type: bool
    ):
        # type: (...) -> None
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

    def assertLieGroupNear(
        self,
        actual,  # type: geo.base.LieGroup
        desired,  # type: geo.base.LieGroup
        places=7,  # type: int
        msg="",  # type: str
        verbose=True,  # type: bool
    ):
        # type: (...) -> None
        """
        Check that two LieGroup elements are close.
        """
        epsilon = 10 ** (-max(9, places + 1))
        # Compute the tangent space pertubation around `actual` that produces `desired`
        local_coordinates = LieGroupOps.local_coordinates(actual, desired, epsilon=epsilon)
        # Compute the identity tangent space pertubation to compare against
        identity = geo.Matrix.zeros(LieGroupOps.tangent_dim(actual), 1)
        return np.testing.assert_almost_equal(
            actual=StorageOps.evalf(local_coordinates),
            desired=StorageOps.to_storage(identity),
            decimal=places,
            err_msg=msg,
            verbose=verbose,
        )

    def compare_or_update(self, path, data):
        # type: (str, str) -> None
        """
        Compare the given data to what is saved in path, OR update the saved data if
        the --update flag was passed to the test.
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
                data,
                expected_data,
                "Data did not match, use --update to check diff and commit if desired.",
            )

    def compare_or_update_directory(self, actual_dir, expected_dir):
        # type: (str, str) -> None
        """
        Check the contents of actual_dir match expected_dir, OR update the expected directory
        if the --update flag was passed to the test.
        """
        logger.debug(
            'Comparing directories: actual="{}", expected="{}"'.format(actual_dir, expected_dir)
        )
        actual_paths = list(python_util.files_in_dir(actual_dir, relative=True))
        expected_paths = list(python_util.files_in_dir(expected_dir, relative=True))

        if not self.UPDATE:
            # If checking, make sure all file paths are the same
            self.assertSequenceEqual(actual_paths, expected_paths)
        else:
            # If updating, remove any expected files not in actual
            for only_in_expected in set(expected_paths).difference(set(actual_paths)):
                os.remove(os.path.join(expected_dir, only_in_expected))

        for path in actual_paths:
            with open(os.path.join(actual_dir, path), "r") as f:
                actual_data = f.read()
            self.compare_or_update(os.path.join(expected_dir, path), actual_data)
