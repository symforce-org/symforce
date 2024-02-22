# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import os
import shutil
import subprocess
from pathlib import Path

from ruff.__main__ import find_ruff_bin

from symforce import typing as T


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


_ruff_path: T.Optional[Path] = None


def _find_ruff() -> Path:
    """
    Find the ruff binary

    `find_ruff_bin` does not work in all environments, for example it does not work on debian when
    things are installed in `/usr/local/bin` and `sysconfig` only returns `/usr/bin`.  Adding
    `shutil.which` should cover most cases, but not all, the better solution would require `ruff`
    putting the binary in `data` like `clang-format` does
    """
    global _ruff_path  # pylint: disable=global-statement

    if _ruff_path is not None:
        return _ruff_path

    try:
        ruff = find_ruff_bin()
    except FileNotFoundError as ex:
        ruff = shutil.which("ruff")
        if ruff is None:
            raise FileNotFoundError("Could not find ruff") from ex

    _ruff_path = ruff

    return ruff


def format_py(file_contents: str, filename: str) -> str:
    """
    Autoformat a given Python file using ruff

    Args:
        filename: A name that this file might have on disk; this does not have to be a real path,
            it's only used for ruff to find the correct style file (by traversing upwards from this
            location)
    """
    result = subprocess.run(
        [_find_ruff(), "format", f"--stdin-filename={filename}", "-"],
        input=file_contents,
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
        [_find_ruff(), "format", dirname],
        check=True,
        # Disable the ruff cache.  This is important for running in a hermetic context like a bazel
        # test, and shouldn't really hurt other use cases.  If it does, we should work around this
        # differently.
        env=dict(os.environ, RUFF_NO_CACHE="true"),
        text=True,
    )
