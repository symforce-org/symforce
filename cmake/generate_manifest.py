#!/usr/bin/env python3

# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
Generate a manifest.json file with build-related information known to CMake at configure time.

Intended to be called from symforce/CMakeLists.txt.

Examples are include paths of libraries we may want to build generated files against, as well as
paths to symenginepy and the lcm-gen executable.  This json file is read by symforce/path_util.py,
and Python users who need the information in the manifest should get it from there.
"""

import argparse
import enum
import json
from pathlib import Path
import re
import typing as T


class AllowedCount(enum.Enum):
    SINGLE = enum.auto()
    MULTIPLE = enum.auto()


REQUIRED_KEYS = (
    ("eigen_include_dirs", AllowedCount.MULTIPLE),
    ("spdlog_include_dirs", AllowedCount.MULTIPLE),
    ("catch2_include_dirs", AllowedCount.MULTIPLE),
    ("symenginepy_install_dir", AllowedCount.SINGLE),
    ("cc_sym_install_dir", AllowedCount.SINGLE),
    ("binary_output_dir", AllowedCount.SINGLE),
)


def parse_cmake_path_list(key: str, cmake_path_list: str) -> T.List[str]:
    """
    Take in a string representing a CMake list of paths, and parse a single path out of it.  If the
    list contains multiple paths, it's required that all but one are generator expressions not
    included in build targets.

    Args:
        cmake_path_list: A CMake list of paths

    Returns:
        The Python list of paths
    """
    # NOTE(aaron): Who knows what happens if someone puts cmake under a directory containing the ';'
    # character? Or a directory that looks like a CMake generator expression? I don't
    maybe_dirs = cmake_path_list.split(";")

    paths = []

    for maybe_dir in maybe_dirs:
        if not maybe_dir:
            # This one was the empty string
            continue

        maybe_generator_match = re.match(r"\$<([A-Za-z_]+):(.+)>", maybe_dir)
        if maybe_generator_match is None:
            # This one was non-empty and not a generator, assume it's a path
            paths.append(maybe_dir)
        else:
            # This one is a generator
            generator_type, path = maybe_generator_match.groups()

            if generator_type == "BUILD_INTERFACE":
                # This one is in the build interface, so keep it
                paths.append(path)
            else:
                # This one was some other generator, skip it
                continue

    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    for key, _ in REQUIRED_KEYS:
        parser.add_argument("--{}".format(key), required=True)
    parser.add_argument(
        "--manifest_path", help="Where to put the generated manifest.json", required=True
    )
    args = parser.parse_args()

    manifest: T.Dict[str, T.Union[str, T.List[str]]] = {}
    for key, allowed_count in REQUIRED_KEYS:
        arg = getattr(args, key)
        path_list = parse_cmake_path_list(key, arg)
        if allowed_count == AllowedCount.SINGLE:
            if len(path_list) != 1:
                raise ValueError("Expected one path for argument {}, got: {}".format(key, arg))
            manifest[key] = path_list[0]
        elif allowed_count == AllowedCount.MULTIPLE:
            if not path_list:
                raise ValueError("Got no paths for {}: {}".format(key, arg))
            manifest[key] = path_list

    Path(args.manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.manifest_path).open("w") as f:
        json.dump(manifest, f)


if __name__ == "__main__":
    main()
