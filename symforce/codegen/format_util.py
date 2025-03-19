# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import os
import shutil
import subprocess
from pathlib import Path

from symforce import typing as T
from symforce.python_util import find_ruff_bin


def format_cpp(file_contents: str, filename: str) -> str:
    """
    Autoformat a given C++ file using clang-format

    Args:
        file_contents: The unformatted contents of the file
        filename: A name that this file might have on disk; this does not have to be a real path,
            it's only used for clang-format to find the correct style file (by traversing upwards
            from this location) and to decide if an include is the corresponding .h file for a .cc
            file that's being formatted (this affects the include order)

    Returns:
        formatted_file_contents (str): The contents of the file after formatting
    """
    try:
        import clang_format

        clang_format_path = str(
            Path(clang_format.__file__).parent / "data" / "bin" / "clang-format"
        )
    except ImportError:
        clang_format_path = "clang-format"

    result = subprocess.run(
        [clang_format_path, f"-assume-filename={filename}"],
        input=file_contents,
        stdout=subprocess.PIPE,
        stderr=None,
        check=True,
        text=True,
    )

    return result.stdout


def format_py(file_contents: str, filename: str) -> str:
    """
    Autoformat a given Python file using ruff

    Args:
        filename: A name that this file might have on disk; this does not have to be a real path,
            it's only used for ruff to find the correct style file (by traversing upwards from this
            location)
    """
    result = subprocess.run(
        [find_ruff_bin(), "format", f"--stdin-filename={filename}", "-"],
        input=file_contents,
        stdout=subprocess.PIPE,
        check=True,
        # Disable the ruff cache.  This is important for running in a hermetic context like a bazel
        # test, and shouldn't really hurt other use cases.  If it does, we should work around this
        # differently.
        env=dict(os.environ, RUFF_NO_CACHE="true"),
        text=True,
    )
    result = subprocess.run(
        [
            find_ruff_bin(),
            "check",
            "--select=I",
            "--fix",
            "--quiet",
            f"--stdin-filename={filename}",
            "-",
        ],
        input=result.stdout,
        stdout=subprocess.PIPE,
        check=True,
        # Disable the ruff cache.  This is important for running in a hermetic context like a bazel
        # test, and shouldn't really hurt other use cases.  If it does, we should work around this
        # differently.
        env=dict(os.environ, RUFF_NO_CACHE="true"),
        text=True,
    )
    return result.stdout


def format_py_dir(dirname: T.Openable) -> None:
    """
    Autoformat python files in a directory (recursively) in-place
    """
    subprocess.run(
        [find_ruff_bin(), "format", dirname],
        check=True,
        # Disable the ruff cache.  This is important for running in a hermetic context like a bazel
        # test, and shouldn't really hurt other use cases.  If it does, we should work around this
        # differently.
        env=dict(os.environ, RUFF_NO_CACHE="true"),
        text=True,
    )


_rustfmt_path: T.Optional[Path] = None


def _find_rustfmt() -> Path:
    """
    Find the rustfmt binary

    """
    global _rustfmt_path  # noqa: PLW0603

    if _rustfmt_path is not None:
        return _rustfmt_path

    rustfmt = shutil.which("rustfmt")
    if rustfmt is None:
        raise FileNotFoundError("Could not find rustfmt")

    # Ignore the type because mypy can't reason about the fact that we just checked that rustfmt
    # is not None.
    _rustfmt_path = Path(rustfmt)
    return _rustfmt_path


def format_rust(file_contents: str, filename: str) -> str:
    """
    Autoformat a given Rust file using rustfmt.

    Args:
        filename: A name that this file might have on disk; this does not have to be a real path,
            it's only used for ruff to find the correct style file (by traversing upwards from this
            location)
    """
    result = subprocess.run(
        [_find_rustfmt()],
        input=file_contents,
        stdout=subprocess.PIPE,
        check=True,
        text=True,
    )
    return result.stdout
