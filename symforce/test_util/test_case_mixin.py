import logging
import os
import sys
import unittest

from symforce import logger
from symforce import python_util
from symforce import types as T


class SymforceTestCaseMixin(object):
    """
    Mixin for SymForce tests, adds useful helpers for code generation
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
            SymforceTestCaseMixin.UPDATE = True
            sys.argv.remove("--update")

        unittest.main()

    def compare_or_update(self, path, data):
        # type: (str, str) -> None
        """
        Compare the given data to what is saved in path, OR update the saved data if
        the --update flag was passed to the test.
        """
        # NOTE(hayk): The output of CSE depends on whether it was run with python 2 or 3,
        # for a currently unknown reason, despite having the same version of sympy. For now
        # only check if python 2 is running.
        if sys.version.startswith("3"):
            return
        if SymforceTestCaseMixin.UPDATE:
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

            T.cast(unittest.TestCase, self).assertMultiLineEqual(
                data,
                expected_data,
                "Data did not match, use --update to check diff and commit if desired.",
            )

    def compare_or_update_file(self, path, new_file):
        # type: (str, str) -> None
        # NOTE(hayk): The output of CSE depends on whether it was run with python 2 or 3,
        # for a currently unknown reason, despite having the same version of sympy. For now
        # only check if python 2 is running.
        if sys.version.startswith("3"):
            return
        with open(new_file) as f:
            code = f.read()
        self.compare_or_update(path, code)

    def compare_or_update_directory(self, actual_dir, expected_dir):
        # type: (str, str) -> None
        """
        Check the contents of actual_dir match expected_dir, OR update the expected directory
        if the --update flag was passed to the test.
        """
        # NOTE(hayk): The output of CSE depends on whether it was run with python 2 or 3,
        # for a currently unknown reason, despite having the same version of sympy. For now
        # only check if python 2 is running.
        if sys.version.startswith("3"):
            return
        logger.debug(
            'Comparing directories: actual="{}", expected="{}"'.format(actual_dir, expected_dir)
        )
        actual_paths = sorted(list(python_util.files_in_dir(actual_dir, relative=True)))
        expected_paths = sorted(list(python_util.files_in_dir(expected_dir, relative=True)))

        if not SymforceTestCaseMixin.UPDATE:
            # If checking, make sure all file paths are the same
            T.cast(unittest.TestCase, self).assertSequenceEqual(actual_paths, expected_paths)
        else:
            # If updating, remove any expected files not in actual
            for only_in_expected in set(expected_paths).difference(set(actual_paths)):
                os.remove(os.path.join(expected_dir, only_in_expected))

        for path in actual_paths:
            self.compare_or_update_file(
                os.path.join(expected_dir, path), os.path.join(actual_dir, path)
            )
