import os
import subprocess

from symforce import logger
from symforce import python_util
from symforce import types as T
from symforce.test_util import TestCase, slow_on_sympy

SYMFORCE_DIR = os.path.dirname(os.path.dirname(__file__)) or "."


class SymforceDocsTest(TestCase):
    """
    Make sure docs can build, as a merge guard.
    """

    @slow_on_sympy
    def test_make_docs(self) -> None:
        try:
            python_util.execute_subprocess(["make", "docs"], cwd=SYMFORCE_DIR)
        except subprocess.CalledProcessError as exc:
            logger.error(exc)
            self.assertTrue(False, "Docs generation failed")


if __name__ == "__main__":
    TestCase.main()
