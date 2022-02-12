# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import os
import subprocess

from symforce import logger
from symforce import python_util
from symforce import typing as T
from symforce.test_util import TestCase, slow_on_sympy

SYMFORCE_DIR = os.path.dirname(os.path.dirname(__file__)) or "."


class SymforceLinterTest(TestCase):
    """
    Make sure linting passes, as a merge guard.
    """

    @slow_on_sympy
    def test_linter(self) -> None:
        try:
            python_util.execute_subprocess(["make", "lint"], cwd=SYMFORCE_DIR)
        except subprocess.CalledProcessError as exc:
            logger.error(exc)
            self.assertTrue(False, "Linter Failed.")


if __name__ == "__main__":
    TestCase.main()
