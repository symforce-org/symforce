#!/usr/bin/env python3

# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
A tool to run a command based on whether the git-hash of the provided path matches the hash in the
stamp file

For builds not in a git repo, we just run the command every time.
"""

from __future__ import print_function

import argparse
import errno
import os
import subprocess
import tempfile
import typing as T
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def tmp_git_index() -> T.Iterator[str]:
    try:
        tmp = tempfile.NamedTemporaryFile(prefix="gitindex", delete=False)
        tmp.close()
        yield tmp.name
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


def get_output(*args: T.Any, **kwargs: T.Any) -> str:
    return subprocess.check_output(*args, **kwargs).decode("utf-8").strip()


# The logic here is fairly complicated, so we have optional debug printing
# 0 = no debug
# 1 = print out the file that triggers the more expensive temporary index update
# 2 = print out the contents of tree_hash and diff_index
DEBUG_GIT_HASHES = int(os.environ.get("DEBUG_GIT_HASHES", "0"))


def _get_hashes(
    relative_path: str, cwd: T.Optional[str] = None, env: T.Optional[T.Mapping[str, str]] = None
) -> T.Tuple[str, str]:
    if relative_path == ".":
        # Getting tree hash of root is not allowed unless it's clear that it's a directory to git.
        relative_path = "./"

    # Get "tree hash" for a subdirectory of a repo. This hash uniquely identifies the contents of
    # the directory.
    tree_hash = get_output(["git", "rev-parse", "HEAD:{}".format(relative_path)], cwd=cwd, env=env)
    # Compares the content and modes of blobs stored in HEAD with the ones in the working tree.
    # Does not always inspect the contents of the changed files unless the files are staged in the
    # current index.
    diff_index = get_output(
        ["git", "diff-index", "--ignore-submodules", "HEAD", "--", relative_path], cwd=cwd, env=env
    )

    if DEBUG_GIT_HASHES > 1:
        print(
            "DEBUG: dir={} tree_hash={} diff_index={}".format(relative_path, tree_hash, diff_index)
        )

    return tree_hash, diff_index


def get_path_git_hash(path_to_check: str, repo_root: T.Optional[str] = None) -> str:
    """
    "hash" check for a path stored under git that combines the
    path's "tree hash" and the output of diff-index.

    We parse the output of diff-index and if git doesn't have a sha1 hash for a
    file (e.g. if the file is in the work tree, but hasn't been staged), we
    re-run our check, but with everything under path_to_check added to a
    temporary git index.

    Changes to untracked files won't affect the hash.
    """
    # Get relative path from possibly absolute path
    relative_path = os.path.relpath(path_to_check, repo_root or os.getcwd())

    tree_hash, diff_index = _get_hashes(relative_path, cwd=repo_root)

    # git will emit 40 zeros as an indication that it hasn't calculated the hash.
    # The file might be unstaged or deleted.
    empty_hash = "0" * 40
    have_unknown_hashes = False
    for diff_line in diff_index.splitlines():
        # See git-diff-files section under https://git-scm.com/docs/git-diff-index
        # :mode_src mode_dst sha1_src sha1_dst status\tpath
        _, _, _, sha1_dst, status_path = diff_line[1:].split(" ")

        # if sha1 hasn't been calculated, this file is probably in the working tree
        if sha1_dst == empty_hash:
            status, path = status_path.split("\t")
            if DEBUG_GIT_HASHES:
                print("DEBUG: triggered temporary index - status={} path={}".format(status, path))
            have_unknown_hashes = True
            break

    # If anything is unstaged we add everything to a temporary git index and
    # get new hashes.
    if have_unknown_hashes:
        with tmp_git_index() as tmp_index_path:
            subprocess.check_call(["cp", ".git/index", tmp_index_path], cwd=repo_root)
            git_env = {"GIT_INDEX_FILE": tmp_index_path}
            get_output(["git", "add", relative_path], cwd=repo_root, env=git_env)

            tree_hash, diff_index = _get_hashes(relative_path, cwd=repo_root, env=git_env)

    return "\n".join([tree_hash, diff_index])


def check_path_git_hash(
    path_to_check: str, stamp_file: str, version: T.Optional[str] = None
) -> T.Tuple[bool, T.Optional[str]]:
    """
    Check the path for whether its contents have changed since the stamp_file was written.

    More details in get_path_git_hash.

    Changes to untracked files won't affect the hash.
    """
    need_to_run = True

    try:
        # Note: this doesn't work for builds not in a git repo.  This also might not work if we're
        # included as a submodule in another git repo?  So if this fails, just always rebuild
        # https://stackoverflow.com/a/957978
        repo_root = get_output(["git", "rev-parse", "--show-toplevel"], cwd=path_to_check)
    except subprocess.CalledProcessError as ex:
        if ex.returncode == 128:  # Code for "not a git repository"
            return (False, None)
        else:
            raise

    current_hash = get_path_git_hash(path_to_check, repo_root=repo_root)

    if version is not None:
        current_hash = "{}\n{}".format(version, current_hash)

    if os.path.exists(stamp_file):
        with open(stamp_file) as f:
            old_stamp = f.read()
        if old_stamp == current_hash:
            need_to_run = False

    return (need_to_run, current_hash)


def write_stamp_file(stamp_file_name: str, stamp_contents: str) -> None:
    # Create the folder if it doesn't already exist
    try:
        os.makedirs(os.path.dirname(stamp_file_name))
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    with open(stamp_file_name, "w") as stampfile:
        stampfile.write(stamp_contents)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether the given path has changed "
        "(according to git), and (re)run the command if necessary"
    )
    parser.add_argument(
        "-p",
        "--path_to_check",
        help="(re)run the command if the git hash of this path"
        "has changed since the last time it was run.",
        required=True,
    )

    parser.add_argument("-s", "--stamp_file", required=True)
    parser.add_argument("-c", "--cmake_stampdir", required=True)
    args = parser.parse_args()

    need_to_run, stamp_contents = check_path_git_hash(args.path_to_check, args.stamp_file)

    if need_to_run:
        for path in Path(args.cmake_stampdir).iterdir():
            # This file is not recreated by cmake on rebuilds and is not a stamp
            if path.name.endswith("-source_dirinfo.txt"):
                continue

            # Newer versions of cmake create empty directories in here at configure time, but
            # they're never populated afaict and are not used as stamps
            # https://github.com/Kitware/CMake/blob/5fbac2bb24250eeeb64e2fb4868dcf976ee29d64/Modules/ExternalProject/mkdirs.cmake.in#L16-L19
            if path.is_dir():
                continue

            print(f"Removing: {path}")
            path.unlink()

        if stamp_contents is not None:
            write_stamp_file(args.stamp_file, stamp_contents)
    else:
        print("{} is up to date".format(args.path_to_check))


if __name__ == "__main__":
    main()
