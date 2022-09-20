# aclint: py2 py3
# mypy: allow-untyped-defs
from __future__ import absolute_import, print_function

import os
import typing as T

from skymarshal import parser, syntax_tree, tokenizer

# Path -> packages map of things we already parsed.
FILE_CACHE = dict()  # type: T.Dict[str, T.Any]


def merge_packages(packages, package_map=None):
    # type: (T.List[syntax_tree.Package], T.Dict[str, syntax_tree.Package]) -> T.Dict[str, syntax_tree.Package]
    """Converts a list of packages to a map.
    NOTE(matt): This makes copies instead of modifying the input packages.
    """
    if package_map is None:
        package_map = {}
    # Collect lcmtypes by package.
    for package in packages:
        if package.name not in package_map:
            # This is a new package. Create a copy so the original isn't modified.
            package_map[package.name] = syntax_tree.Package(
                name=package.name, type_definitions=list(package.type_definitions.values())
            )
        else:
            # This package already exists: add the new structs and enums to the existing
            # package object.
            package_map[package.name].extend_with_package(package)
    return package_map


def find_lcmtypes_dirs(root_path, excluded_paths=None):
    remaining_excluded_paths = set(excluded_paths or [])
    for dirpath, dirnames, _ in os.walk(root_path, topdown=True, followlinks=False):
        # directory path relative to root
        rel_dir = os.path.relpath(dirpath, root_path)

        if rel_dir in remaining_excluded_paths:
            remaining_excluded_paths.remove(rel_dir)
            dirnames[:] = []
            continue

        if os.path.basename(dirpath) == "lcmtypes":
            yield rel_dir
            # don't recurse into lcmtypes directories
            dirnames[:] = []


def _flatten_paths(lcmtypes_paths):
    # type: (T.Iterable[str]) -> T.Iterable[str]
    for path in lcmtypes_paths:
        if not os.path.exists(path):
            continue

        if os.path.isdir(path):
            for fname in os.listdir(path):
                if fname.endswith(".lcm"):
                    yield os.path.join(path, fname)
        elif path.endswith(".lcm"):
            yield path


def parse_lcmtypes(
    lcmtypes_paths,  # type: T.Iterable[str]
    verbose=False,  # type: bool
    print_debug_tokens=False,  # type: bool
    cache_parser=False,  # type: bool
    allow_unknown_notations=False,  # type: bool
):
    # type: (...) -> T.Dict[str, syntax_tree.Package]
    """
    Parse LCM definitions and assemble a map from package to syntax tree nodes.

    :param lcmtypes_paths: Iterable of .lcm file paths, or paths to folders of .lcm files, to parse.
    :param print_debug_tokens: If true, print debug info.
    :param cache_parser: If true, cache YACC parser across each package
    :return: Map from package name to syntax_tree.Package.
    """
    package_map = {}  # type: T.Dict[str, syntax_tree.Package]

    for path in _flatten_paths(lcmtypes_paths):
        if path in FILE_CACHE:
            # Get the original list from the cache.
            packages = FILE_CACHE[path]
        else:
            with open(path) as src_file:
                src = src_file.read()

            if print_debug_tokens:
                tokenizer.debug_tokens(src)
            if verbose:
                print(path)
            packages = parser.lcmparse(
                src,
                verbose=verbose,
                cache=cache_parser,
                debug_src_path=path,
                allow_unknown_notations=allow_unknown_notations,
            )

            # Cache the raw package list for this path.
            FILE_CACHE[path] = packages

        # Copy the packages into the map
        merge_packages(packages, package_map)

    return package_map
