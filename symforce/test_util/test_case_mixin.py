# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import difflib
import logging
import os
import re
import sys
import tempfile
import unittest

import numpy as np

import symforce.symbolic as sf
from symforce import logger
from symforce import python_util
from symforce import typing as T
from symforce.codegen import codegen_config
from symforce.ops import LieGroupOps
from symforce.ops import StorageOps
from symforce.ops import interfaces


class SymforceTestCaseMixin(unittest.TestCase):
    """
    Mixin for SymForce tests, adds useful helpers for code generation
    """

    LieGroupOpsType = T.Union[interfaces.LieGroup, sf.Scalar]

    # Set by the --update flag to tell tests that compare against some saved
    # data to update that data instead of failing
    UPDATE = False

    KEEP_PATHS = [
        r".*/__pycache__/.*",
        r".*\.pyc",
    ]

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        # Registers assertArrayEqual with python unittest TestCase such that we use numpy array
        # comparison functions rather than the "==" operator, which throws an error for ndarrays
        self.addTypeEqualityFunc(np.ndarray, SymforceTestCaseMixin.assertArrayEqual)

    @staticmethod
    def main(*args: T.Any, **kwargs: T.Any) -> None:
        """
        Call this to run all tests in scope.
        """
        # Sneak through options to expose to tests
        if "--update" in sys.argv:
            SymforceTestCaseMixin.UPDATE = True
            sys.argv.remove("--update")

        unittest.main(*args, **kwargs)

    @staticmethod
    def assertStorageNear(
        actual: T.Any, desired: T.Any, *, places: int = 7, msg: str = "", verbose: bool = True
    ) -> None:
        """
        Check that two elements are close. Handles sequences, scalars, and geometry types
        using StorageOps.
        """
        return np.testing.assert_almost_equal(
            actual=np.array(StorageOps.evalf(StorageOps.to_storage(actual)), dtype=np.double),
            desired=np.array(StorageOps.evalf(StorageOps.to_storage(desired)), dtype=np.double),
            decimal=places,
            err_msg=msg,
            verbose=verbose,
        )

    @staticmethod
    def assertLieGroupNear(
        actual: LieGroupOpsType,
        desired: LieGroupOpsType,
        *,
        places: int = 7,
        msg: str = "",
        verbose: bool = True,
    ) -> None:
        """
        Check that two LieGroup elements are close.
        """
        epsilon = 10 ** (-max(9, places + 1))
        # Compute the tangent space pertubation around `actual` that produces `desired`
        local_coordinates = LieGroupOps.local_coordinates(actual, desired, epsilon=epsilon)
        # Compute the identity tangent space pertubation to compare against
        identity = sf.Matrix.zeros(LieGroupOps.tangent_dim(actual), 1)
        return np.testing.assert_almost_equal(
            actual=StorageOps.evalf(local_coordinates),
            desired=StorageOps.to_storage(identity),
            decimal=places,
            err_msg=msg,
            verbose=verbose,
        )

    @staticmethod
    def assertArrayEqual(actual: T.ArrayElement, desired: T.ArrayElement, msg: str = "") -> None:
        """
        Called by unittest base class when comparing ndarrays when "assertEqual" is called.
        By default, "assertEqual" uses the "==" operator, which is not implemented for ndarrays.
        """
        return np.testing.assert_array_equal(actual, desired, err_msg=msg)

    def assertNotEqual(self, first: T.Any, second: T.Any, msg: str = "") -> None:
        """
        Overrides unittest.assertNotEqual to handle ndarrays separately. "assertNotEqual"
        uses the "!=" operator, but this is not implemented for ndarrays. Instead, we check that
        np.testing.assert_array_equal raises an assertion error, as numpy testing does not provide
        a assert_array_not_equal function.

        Note that assertNotEqual does not work like assertEqual in unittest. Rather than
        allowing you to register a custom equality evaluator (e.g. with `addTypeEqualityFunc()`),
        assertNotEqual assumes the "!=" can be used with the arguments regardless of type.
        """
        if isinstance(first, np.ndarray):
            return np.testing.assert_raises(
                AssertionError, np.testing.assert_array_equal, first, second, msg
            )
        else:
            return super().assertNotEqual(first, second, msg)

    def make_output_dir(self, prefix: str, directory: str = "/tmp") -> str:
        """
        Create a temporary output directory, which will be automatically removed (regardless of
        exceptions) on shutdown, unless logger.level is DEBUG

        Args:
            prefix: The prefix for the directory name - a random unique identifier is added to this
            dir: Location of the output directory. Defaults to "/tmp".

        Returns:
            str: The absolute path to the created output directory
        """
        output_dir = tempfile.mkdtemp(prefix=prefix, dir=directory)
        logger.debug(f"Creating temp directory: {output_dir}")
        self.output_dirs.append(output_dir)
        return output_dir

    def setUp(self) -> None:
        """
        Creates list of temporary directories that will be removed before shutdown (unless debug
        mode is on)
        """
        super().setUp()

        # Set to fail on default epsilon == 0
        codegen_config.DEFAULT_ZERO_EPSILON_BEHAVIOR = codegen_config.ZeroEpsilonBehavior.FAIL

        # Storage for temporary output directories
        self.output_dirs: T.List[str] = []

    def tearDown(self) -> None:
        """
        Removes temporary output directories (unless debug mode is on)
        """
        super().tearDown()
        if logger.level != logging.DEBUG:
            for output_dir in self.output_dirs:
                python_util.remove_if_exists(output_dir)

    def compare_or_update(self, path: T.Openable, data: str) -> None:
        """
        Compare the given data to what is saved in path, OR update the saved data if
        the --update flag was passed to the test.
        """
        if SymforceTestCaseMixin.UPDATE:
            logger.debug(f'Updating data at: "{path}"')

            dirname = os.path.dirname(path)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)

            with open(path, "w") as f:
                f.write(data)
        else:
            logger.debug(f'Comparing data at: "{path}"')
            with open(path) as f:
                expected_data = f.read()

            if data != expected_data:
                diff = difflib.unified_diff(
                    expected_data.splitlines(), data.splitlines(), "expected", "got", lineterm=""
                )
                self.fail(
                    "\n"
                    + "\n".join(diff)
                    + f"\n\n{80*'='}\nData did not match for file {path}, see diff above.  Use "
                    "`--update` to write the changes to the working directory and commit if desired"
                )

    def compare_or_update_file(self, path: T.Openable, new_file: T.Openable) -> None:
        with open(new_file) as f:
            code = f.read()
        self.compare_or_update(path, code)

    def _filtered_paths_in_dir(self, directory: T.Openable) -> T.List[str]:
        """
        Find the list of paths in a directory not in KEEP_PATHS, recursively.  The result is in
        sorted order
        """
        keep_regex = re.compile("|".join(self.KEEP_PATHS))

        files_in_dir = python_util.files_in_dir(directory, relative=True)
        return sorted(path for path in files_in_dir if not re.match(keep_regex, path))

    def compare_or_update_directory(self, actual_dir: T.Openable, expected_dir: T.Openable) -> None:
        """
        Check the contents of actual_dir match expected_dir, OR update the expected directory
        if the --update flag was passed to the test.
        """
        logger.debug(f'Comparing directories: actual="{actual_dir}", expected="{expected_dir}"')

        actual_paths = self._filtered_paths_in_dir(actual_dir)
        expected_paths = self._filtered_paths_in_dir(expected_dir)

        if not SymforceTestCaseMixin.UPDATE:
            # If checking, make sure all file paths are the same
            self.assertSequenceEqual(actual_paths, expected_paths)
        else:
            # If updating, remove any expected files not in actual
            for only_in_expected in set(expected_paths).difference(set(actual_paths)):
                os.remove(os.path.join(expected_dir, only_in_expected))

        for path in actual_paths:
            self.compare_or_update_file(
                os.path.join(expected_dir, path), os.path.join(actual_dir, path)
            )
