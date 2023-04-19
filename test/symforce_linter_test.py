# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import os
import subprocess
import sys
import unittest
from pathlib import Path

from symforce import logger
from symforce import python_util
from symforce.test_util import TestCase
from symforce.test_util import slow_on_sympy

SYMFORCE_DIR = Path(__file__).parent.parent


class SymforceLinterTest(TestCase):
    """
    Make sure linting passes, as a merge guard.
    """

    @slow_on_sympy
    @unittest.skipIf(
        sys.version_info[:3] >= (3, 10, 7),
        """
        Mypy fails on Python 3.10.7 because of this bug, which is fixed in mypy 0.981:
        https://github.com/python/mypy/issues/13627
        """,
    )
    def test_linter(self) -> None:
        try:
            python_util.execute_subprocess(
                ["make", "lint"], cwd=SYMFORCE_DIR, env=dict(os.environ, PYTHON=sys.executable)
            )
        except subprocess.CalledProcessError as exc:
            logger.error(exc)
            self.fail("Linter Failed.")


if __name__ == "__main__":
    TestCase.main()
