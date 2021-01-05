import os
import subprocess

from symforce import logger
from symforce import python_util
from symforce.test_util import TestCase

SYMFORCE_DIR = os.path.dirname(os.path.dirname(__file__)) or "."


class SymforceMakeTestSymEngine(TestCase):
    """
    Run `make test_symengine`, which runs all python tests on SymEngine
    """

    def test_with_make(self) -> None:
        try:
            python_util.execute_subprocess(["make", "test_symengine"], cwd=SYMFORCE_DIR)
        except subprocess.CalledProcessError as exc:
            logger.error(exc)
            self.assertTrue(False, "make test_symengine failed")


if __name__ == "__main__":
    TestCase.main()
