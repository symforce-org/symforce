# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import os
from pathlib import Path
import sys

from symforce.test_util import TestCase
from symforce import python_util
from symforce import typing as T

SYMFORCE_DIR = Path(__file__).resolve().parent.parent


class SymforceRequirementsTest(TestCase):
    """
    Tests pip requirements
    """

    # Pass the --upgrade flag to piptools?
    _PIPTOOLS_UPGRADE = False

    @staticmethod
    def main(*args: T.Any, **kwargs: T.Any) -> None:
        """
        Call this to run all tests in scope.
        """
        if "--piptools_upgrade" in sys.argv:
            SymforceRequirementsTest._PIPTOOLS_UPGRADE = True
            sys.argv.remove("--piptools_upgrade")

        TestCase.main(*args, **kwargs)

    def test_requirements(self) -> None:
        output_dir = Path(self.make_output_dir("sf_requirements_test_"))

        output_requirements_file = output_dir / "dev_requirements.txt"
        symforce_requirements_file = SYMFORCE_DIR / "dev_requirements.txt"

        local_requirements_map = {
            f"skymarshal @ file://localhost/{SYMFORCE_DIR}/third_party/skymarshal": "file:./third_party/skymarshal",
            f"symforce-sym @ file://localhost/{SYMFORCE_DIR}/gen/python": "file:./gen/python",
        }

        # Copy the symforce requirements file into the temp directory
        # This is necessary so piptools has current versions of the packages already in the list
        if symforce_requirements_file.exists():
            requirements_contents = symforce_requirements_file.read_text()

            # Rewrite local paths to absolute paths
            # Pip isn't technically supposed to support relative paths, but it does and they're much
            # nicer for this use case
            # https://stackoverflow.com/a/64809439/2791611
            for key, value in local_requirements_map.items():
                requirements_contents = requirements_contents.replace(value, key)

            output_requirements_file.write_text(requirements_contents)

        maybe_piptools_upgrade = ["--upgrade"] if self._PIPTOOLS_UPGRADE else []

        python_util.execute_subprocess(
            [
                sys.executable,
                "-m",
                "piptools",
                "compile",
                f"--output-file={output_requirements_file}",
                "--index-url=https://pypi.python.org/simple",
                "--allow-unsafe",
                "--extra=dev",
                "--extra=_setup",
            ]
            + maybe_piptools_upgrade,
            cwd=SYMFORCE_DIR,
            env=dict(
                os.environ,
                # Compile command to put in the header of dev_requirements.txt
                CUSTOM_COMPILE_COMMAND="python test/symforce_requirements_test.py --update",
            ),
        )

        # Reverse path rewrite back to relative paths
        requirements_contents = output_requirements_file.read_text()
        for key, value in local_requirements_map.items():
            requirements_contents = requirements_contents.replace(key, value)
        output_requirements_file.write_text(requirements_contents)

        self.compare_or_update_file(symforce_requirements_file, output_requirements_file)


if __name__ == "__main__":
    SymforceRequirementsTest.main()
