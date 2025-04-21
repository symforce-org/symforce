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
from symforce.test_util import requires_source_build
from symforce.test_util import symengine_only

SYMFORCE_DIR = Path(__file__).parent.parent


class SymforceLinterTest(TestCase):
    """
    Make sure linting passes, as a merge guard.
    """

    @requires_source_build
    @symengine_only
    def test_linter(self) -> None:
        try:
            asyncio.run(
                python_util.execute_subprocess(
                    ["make", "lint", "--keep-going"],
                    cwd=SYMFORCE_DIR,
                    env=dict(os.environ, PYTHON=sys.executable),
                )
            )
        except subprocess.CalledProcessError as exc:
            logger.error(exc)
            self.fail("Linter Failed.")


if __name__ == "__main__":
    TestCase.main()
