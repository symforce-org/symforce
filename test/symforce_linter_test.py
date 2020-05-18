# mypy: disallow-untyped-defs

import os
import subprocess

from symforce import logger
from symforce import python_util
from symforce import types as T
from symforce.test_util import TestCase

SYMFORCE_DIR = os.path.dirname(os.path.dirname(__file__)) or "."


class SymforceLinterTest(TestCase):
    """
    Make sure linting passes, as a merge guard.
    """

    def test_linter(self):
        # type: () -> None
        try:
            python_util.execute_subprocess(["make", "lint"], cwd=SYMFORCE_DIR)
        except subprocess.CalledProcessError as exc:
            logger.error(exc)
            self.assertTrue(False, "Linter Failed.")


if __name__ == "__main__":
    TestCase.main()
