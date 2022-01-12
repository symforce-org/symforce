import logging
import os
import re
import sys
import tempfile
import unittest

import symforce
from symforce import logger
from symforce import python_util
from symforce import typing as T


class SymforceTestCaseMixin(unittest.TestCase):
    """
    Mixin for SymForce tests, adds useful helpers for code generation
    """

    # Set by the --update flag to tell tests that compare against some saved
    # data to update that data instead of failing
    UPDATE = False

    KEEP_PATHS = [
        r".*/__pycache__/.*",
        r".*\.pyc",
    ]

    @staticmethod
    def main() -> None:
        """
        Call this to run all tests in scope.
        """
        # Sneak through options to expose to tests
        if "--update" in sys.argv:
            SymforceTestCaseMixin.UPDATE = True
            sys.argv.remove("--update")

        unittest.main()

    def make_output_dir(self, prefix: str, dir: str = "/tmp") -> str:
        """
        Create a temporary output directory, which will be automatically removed (regardless of
        exceptions) on shutdown, unless logger.level is DEBUG

        Args:
            prefix: The prefix for the directory name - a random unique identifier is added to this
            dir: Location of the output directory. Defaults to "/tmp".

        Returns:
            str: The absolute path to the created output directory
        """
        output_dir = tempfile.mkdtemp(prefix=prefix, dir=dir)
        logger.debug(f"Creating temp directory: {output_dir}")
        self.output_dirs.append(output_dir)
        return output_dir

    def setUp(self) -> None:
        """
        Creates list of temporary directories that will be removed before shutdown (unless debug
        mode is on)
        """
        super().setUp()
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
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            with open(path, "w") as f:
                f.write(data)
        else:
            logger.debug(f'Comparing data at: "{path}"')
            with open(path) as f:
                expected_data = f.read()

            self.assertMultiLineEqual(
                data,
                expected_data,
                "Data did not match, use --update to check diff and commit if desired.",
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
