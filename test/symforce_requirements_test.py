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
from symforce.test_util import requires_source_build
from symforce.test_util import sympy_only


class SymforceRequirementsTest(TestCase):
    """
    Generates pinned pip requirements for all python versions, and tests that solving for
    requirements gives the same result as the checked-in requirements files.

    To re-solve for the requirements, pass --update.  To also upgrade all packages to latest
    (subject to given constraints), also pass --uv_arg=--upgrade.  Or upgrade one package to
    latest with --uv_arg=--upgrade-package=<something>, etc.
    """

    # Extra args for uv
    _UV_ARGS: T.List[str] = []

    @staticmethod
    def main(*args: T.Any, **kwargs: T.Any) -> None:
        """
        Call this to run all tests in scope.
        """
        argv: T.List[str] = []
        for arg in sys.argv:
            if arg.startswith("--uv_arg="):
                SymforceRequirementsTest._UV_ARGS.append(arg[len("--uv_arg=") :])
            else:
                argv.append(arg)
        sys.argv = argv

        TestCase.main(*args, **kwargs)

    def check_dev_requirements_for_version(self, version: int) -> None:
        output_dir = self.make_output_dir("sf_requirements_test_dev_")

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

        # Do not use the cache if we have no custom args (for hermeticity)
        extra_uv_args = self._UV_ARGS or ["--no-cache"]

        asyncio.run(
            python_util.execute_subprocess(
                [
                    python_util.find_uv_bin(),
                    "pip",
                    "compile",
                    "--all-extras",
                    f"--output-file={output_requirements_file}",
                    f"--python-version=3.{version}",
                    "pyproject.toml",
                ]
                + extra_uv_args,
                cwd=path_util.symforce_root(),
                env=dict(
                    os.environ,
                    # Compile command to put in the header of requirements.txt
                    UV_CUSTOM_COMPILE_COMMAND="python test/symforce_requirements_test.py --update",
                    # Don't actually deduce versions for these, we don't need them
                    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SYMFORCE="0.1.0",
                    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SYMFORCE_SYM="0.1.0",
                    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SKYMARSHAL="0.1.0",
                ),
            )
        )

        # Reverse path rewrite back to relative paths
        requirements_contents = output_requirements_file.read_text()
        for key, value in local_requirements_map.items():
            requirements_contents = re.sub(key.format(".*"), value, requirements_contents)

        output_requirements_file.write_text(requirements_contents)

        self.compare_or_update_file(
            path_util.symforce_data_root(__file__) / f"requirements_dev_py3{version}.txt",
            output_requirements_file,
        )

    @requires_source_build
    @sympy_only
    def test_dev_requirements_py38(self) -> None:
        self.check_dev_requirements_for_version(8)

    @requires_source_build
    @sympy_only
    def test_dev_requirements_py39(self) -> None:
        self.check_dev_requirements_for_version(9)

    @requires_source_build
    @sympy_only
    def test_dev_requirements_py310(self) -> None:
        self.check_dev_requirements_for_version(10)

    @requires_source_build
    @sympy_only
    def test_dev_requirements_py311(self) -> None:
        self.check_dev_requirements_for_version(11)

    @requires_source_build
    @sympy_only
    def test_dev_requirements_py312(self) -> None:
        self.check_dev_requirements_for_version(12)

    @requires_source_build
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
                    python_util.find_uv_bin(),
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
                    # Don't actually deduce versions for these, we don't need them
                    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SYMFORCE="0.1.0",
                    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SYMFORCE_SYM="0.1.0",
                    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SKYMARSHAL="0.1.0",
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
            path_util.symforce_data_root(__file__) / "requirements_build.txt",
            output_requirements_file,
        )


if __name__ == "__main__":
    SymforceRequirementsTest.main()
