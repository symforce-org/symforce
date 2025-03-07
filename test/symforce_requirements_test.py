# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import asyncio
import os
import re
import sys

from symforce import path_util
from symforce import python_util
from symforce import typing as T
from symforce.test_util import TestCase
from symforce.test_util import sympy_only


class SymforceRequirementsTest(TestCase):
    """
    Generates pinned pip requirements for all python versions, and tests that solving for
    requirements gives the same result as the checked-in requirements files.

    Running this on a given version of python only checks or updates requirements for the running
    python version.  CI runs this test on all supported versions of python.

    To update the requirements files, trigger the solve_requirements GitHub Action, either on main
    to solve with no changes to dependencies (in e.g. setup.py or pyproject.toml), or on a branch
    with changes to dependencies.  The action will create a PR against the branch it was triggered
    on with the solved requirements files.
    """

    # Pass the --upgrade flag to uv compile?
    _UV_UPGRADE = False

    @staticmethod
    def main(*args: T.Any, **kwargs: T.Any) -> None:
        """
        Call this to run all tests in scope.
        """
        if "--uv_upgrade" in sys.argv:
            SymforceRequirementsTest._UV_UPGRADE = True
            sys.argv.remove("--uv_upgrade")

        TestCase.main(*args, **kwargs)

    @sympy_only
    def test_dev_requirements(self) -> None:
        output_dir = self.make_output_dir("sf_requirements_test_dev_")

        version = sys.version_info.minor

        output_requirements_file = output_dir / f"requirements_dev_py3{version}.txt"
        symforce_requirements_file = (
            path_util.symforce_root() / f"requirements_dev_py3{version}.txt"
        )

        local_requirements_map = {
            "skymarshal @ file://{}/third_party/skymarshal": "file:./third_party/skymarshal",
            "symforce-sym @ file://{}/gen/python": "file:./gen/python",
        }

        # Copy the symforce requirements file into the temp directory
        # This is necessary so uv has current versions of the packages already in the list
        if symforce_requirements_file.exists():
            requirements_contents = symforce_requirements_file.read_text()

            # Rewrite local paths to absolute paths
            # Pip isn't technically supposed to support relative paths, but it does and they're much
            # nicer for this use case
            # https://stackoverflow.com/a/64809439/2791611
            for key, value in local_requirements_map.items():
                requirements_contents = requirements_contents.replace(
                    value, key.format(path_util.symforce_root())
                )

            output_requirements_file.write_text(requirements_contents)

        # Do not use the cache if not upgrading (for hermeticity)
        maybe_uv_upgrade = ["--upgrade"] if self._UV_UPGRADE else ["--no-cache"]

        asyncio.run(
            python_util.execute_subprocess(
                [
                    "uv",
                    "pip",
                    "compile",
                    "--all-extras",
                    f"--output-file={output_requirements_file}",
                    "pyproject.toml",
                ]
                + maybe_uv_upgrade,
                cwd=path_util.symforce_root(),
                env=dict(
                    os.environ,
                    # Compile command to put in the header of requirements.txt
                    UV_CUSTOM_COMPILE_COMMAND="python test/symforce_requirements_test.py --update",
                ),
            )
        )

        # Reverse path rewrite back to relative paths
        requirements_contents = output_requirements_file.read_text()
        for key, value in local_requirements_map.items():
            requirements_contents = re.sub(key.format(".*"), value, requirements_contents)

        output_requirements_file.write_text(requirements_contents)

        self.compare_or_update_file(
            path_util.symforce_data_root() / f"requirements_dev_py3{version}.txt",
            output_requirements_file,
        )

    @sympy_only
    def test_build_requirements(self) -> None:
        """
        Generate requirements_build.txt
        """
        output_dir = self.make_output_dir("sf_requirements_test_build_")
        output_requirements_file = output_dir / "requirements_build.txt"

        local_requirements_map = {
            "skymarshal @ file://{}/third_party/skymarshal": "file:./third_party/skymarshal",
            "symforce-sym @ file://{}/gen/python": "file:./gen/python",
        }

        asyncio.run(
            python_util.execute_subprocess(
                [
                    "uv",
                    "pip",
                    "compile",
                    "--no-deps",
                    "--extra=setup",
                    "--no-cache",
                    f"--output-file={output_requirements_file}",
                    "pyproject.toml",
                ],
                cwd=path_util.symforce_root(),
                env=dict(
                    os.environ,
                    # Compile command to put in the header of requirements.txt
                    UV_CUSTOM_COMPILE_COMMAND="python test/symforce_requirements_test.py --update",
                ),
            )
        )

        # Reverse path rewrite back to relative paths
        requirements_contents = output_requirements_file.read_text()
        for key, value in local_requirements_map.items():
            requirements_contents = re.sub(key.format(".*"), value, requirements_contents)

        # Strip version numbers
        requirements_contents = re.sub(
            r"==[+-\.a-zA-Z0-9]+$", "", requirements_contents, flags=re.MULTILINE
        )

        output_requirements_file.write_text(requirements_contents)

        self.compare_or_update_file(
            path_util.symforce_data_root() / "requirements_build.txt",
            output_requirements_file,
        )


if __name__ == "__main__":
    SymforceRequirementsTest.main()
