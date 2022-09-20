# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import os
import subprocess
import sys

from symforce import logger
from symforce import python_util
from symforce import typing as T
from symforce.test_util import TestCase, slow_on_sympy

SYMFORCE_DIR = os.path.dirname(os.path.dirname(__file__)) or "."


class SymforceDocsTest(TestCase):
    """
    Make sure docs can build, as a merge guard.
    """

    @slow_on_sympy
    def test_make_docs(self) -> None:
        # This test is occasionally flaky (the jupyter kernel randomly becomes unresponsive?), so
        # retry a couple times
        RETRIES = 3

        success = False
        for _ in range(RETRIES):
            try:
                python_util.execute_subprocess(
                    ["make", "docs"], cwd=SYMFORCE_DIR, env=dict(os.environ, PYTHON=sys.executable)
                )
            except subprocess.CalledProcessError as exc:
                logger.error(exc)
            else:
                success = True
                break

        self.assertTrue(success, "Docs generation failed")


if __name__ == "__main__":
    TestCase.main()
