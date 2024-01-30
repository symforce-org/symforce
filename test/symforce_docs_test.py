# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import asyncio
import os
import subprocess
import sys
from pathlib import Path

from symforce import logger
from symforce import python_util
from symforce.test_util import TestCase
from symforce.test_util import slow_on_sympy

SYMFORCE_DIR = Path(__file__).parent.parent


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
                asyncio.run(
                    python_util.execute_subprocess(
                        ["make", "docs"],
                        cwd=SYMFORCE_DIR,
                        env=dict(os.environ, PYTHON=sys.executable),
                        log_stdout=False,
                    )
                )
            except subprocess.CalledProcessError as exc:
                logger.error(exc)
            else:
                success = True
                break

        self.assertTrue(success, "Docs generation failed")


if __name__ == "__main__":
    TestCase.main()
