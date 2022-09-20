# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import copy
import os
import pathlib

import black

from symforce import python_util
from symforce import typing as T

# TODO(aaron): Put this in a pyproject.toml and fetch from there
BLACK_FILE_MODE = black.FileMode(line_length=100)


def format_cpp(file_contents: str, filename: str) -> str:
    """
    Autoformat a given C++ file using clang-format

    Args:
        file_contents (str): The unformatted contents of the file
        filename (str): A name that this file might have on disk; this does not have to be a real
            path, it's only used for clang-format to find the correct style file (by traversing
            upwards from this location) and to decide if an include is the corresponding .h file
            for a .cc file that's being formatted (this affects the include order)

    Returns:
        formatted_file_contents (str): The contents of the file after formatting
    """
    try:
        import clang_format  # type: ignore[import]

        clang_format_path = str(
            pathlib.Path(clang_format.__file__).parent / "data" / "bin" / "clang-format"
        )
    except ImportError:
        clang_format_path = "clang-format"

    formatted_file_contents = python_util.execute_subprocess(
        [clang_format_path, f"-assume-filename={filename}"],
        stdin_data=file_contents,
        log_stdout=False,
    )

    return formatted_file_contents


def format_py(file_contents: str) -> str:
    """
    Autoformat a given Python file using black
    """
    return black.format_str(file_contents, mode=BLACK_FILE_MODE)


def format_pyi(file_contents: str) -> str:
    """
    Autoformat a given Python interface file using black
    """
    mode = copy.copy(BLACK_FILE_MODE)
    mode.is_pyi = True
    return black.format_str(file_contents, mode=mode)


def format_py_dir(dirname: T.Openable) -> None:
    """
    Autoformat python files in a directory (recursively) in-place
    """
    for root, _, files in os.walk(dirname):
        for filename in files:
            if filename.endswith(".py"):
                black.format_file_in_place(
                    pathlib.Path(os.path.join(dirname, root, filename)),
                    fast=True,
                    mode=BLACK_FILE_MODE,
                    write_back=black.WriteBack.YES,
                )
