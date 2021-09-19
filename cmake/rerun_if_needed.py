#!/usr/bin/python3
"""
A tool to run a command based on one of the following checks:
 - the git-hash of the provided path matches the hash in the stamp file
 - the provided commit is an ancestor of the commit in the stamp file
"""
from __future__ import print_function
import argparse
import errno
import glob
import hashlib
import os
import subprocess
import sys
import tempfile
import typing as T

from contextlib import contextmanager


@contextmanager
def tmp_git_index():
    # type: (...) -> T.Iterator[str]
    try:
        tmp = tempfile.NamedTemporaryFile(prefix="gitindex", delete=False)
        tmp.close()
        yield tmp.name
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


def get_output(*args, **kwargs):
    # type: (T.Any, T.Any) -> str
    return subprocess.check_output(*args, **kwargs).decode("utf-8").strip()


def md5sum(files, relative_to=None):
    # type: (T.Iterable[str], str) -> str
    """
    Inspired by the md5sum unix command.
    Computes a newline separated list of hashes with their associated
    filenames.  Unlike the md5sum shell command, this skips directories.

    if relative_to is provided, writes paths relative to provided path
    """
    stamps = []
    for filename in files:
        if os.path.isdir(filename):
            continue
        with open(filename, "rb") as infile:
            hasher = hashlib.md5()
            hasher.update(infile.read())
            if relative_to is None:
                path = filename
            else:
                path = os.path.relpath(filename, relative_to)
            stamps.append("{}: {}".format(hasher.hexdigest(), path))
    return "\n".join(stamps)


def check_files_digest(files, stamp_file, relative_to=None):
    # type: (T.Sequence[str], str, str) -> T.Tuple[bool, T.Optional[str]]
    """
    Check a set of files for modifications since stamp_file was written.
    The stamp includes a full manifest of matched files and their content hashes.

    if relative_to is provided, paths are output relative to provided path
    """
    files_changed = True
    try:
        current_stamp = md5sum(sorted(set(files)), relative_to=relative_to)
    except IOError:
        # If any file is missing or unreadable, we definitely need to run
        return (True, None)

    if os.path.exists(stamp_file):
        with open(stamp_file) as f:
            old_stamp = f.read()
        if old_stamp == current_stamp:
            files_changed = False

    return (files_changed, current_stamp)


def check_globbed_files_digest(globs_to_check, stamp_file):
    # type: (T.Iterable[str], str) -> T.Tuple[bool, T.Optional[str]]
    """
    Expand the given unix globs, and check for modifications since stamp_file was written.
    The stamp includes a full manifest of matched files and their content hashes.
    """
    files = []
    for glob_to_check in globs_to_check:
        files += [f for f in glob.glob(glob_to_check)]

    return check_files_digest(files, stamp_file)


# The logic here is fairly complicated, so we have optional debug printing
# 0 = no debug
# 1 = print out the file that triggers the more expensive temporary index update
# 2 = print out the contents of tree_hash and diff_index
DEBUG_GIT_HASHES = int(os.environ.get("DEBUG_GIT_HASHES", "0"))


def _get_hashes(relative_path, cwd=None, env=None):
    # type: (str, str, T.Mapping[str, str]) -> T.Tuple[str, str]
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


def get_path_git_hash(path_to_check, repo_root=None):
    # type: (str, str) -> str
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


def check_path_git_hash(path_to_check, stamp_file, version=None, repo_root=None):
    # type: (str, str, str, str) -> T.Tuple[bool, str]
    """
    Check the path for whether its contents have changed since the stamp_file was written.

    More details in get_path_git_hash.

    Changes to untracked files won't affect the hash.
    """
    need_to_run = True
    current_hash = get_path_git_hash(path_to_check, repo_root=repo_root)

    if version is not None:
        current_hash = "{}\n{}".format(version, current_hash)

    if os.path.exists(stamp_file):
        with open(stamp_file) as f:
            old_stamp = f.read()
        if old_stamp == current_hash:
            need_to_run = False

    return (need_to_run, current_hash)


def check_if_commit_is_ancestor(ancestor_commit, stamp_file):
    # type: (str, str) -> T.Tuple[bool, str]
    """
    Check whether the passed in commit is an ancestor of the commit in the stamp_file
    """
    need_to_run = True
    if os.path.exists(stamp_file):
        with open(stamp_file) as f:
            previous_run_commit = f.read()
        need_to_run = bool(
            subprocess.call(
                ["git", "merge-base", "--is-ancestor", ancestor_commit, previous_run_commit]
            )
        )

    current_commit = get_output(["git", "rev-parse", "HEAD"])
    return (need_to_run, current_commit)


def write_stamp_file(stamp_file_name, stamp_contents):
    # type: (str, str) -> None
    # Create the folder if it doesn't already exist
    try:
        os.makedirs(os.path.dirname(stamp_file_name))
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    with open(stamp_file_name, "w") as stampfile:
        stampfile.write(stamp_contents)


def main():
    # type: () -> None
    parser = argparse.ArgumentParser(
        description="Check whether the given path has changed "
        "(according to git), and (re)run the command if necessary"
    )
    to_check_group = parser.add_mutually_exclusive_group(required=True)
    to_check_group.add_argument(
        "-p",
        "--path_to_check",
        help="(re)run the command if the git hash of this path"
        "has changed since the last time it was run.",
    )
    to_check_group.add_argument(
        "-a",
        "--ancestor_commit",
        help="(re)run the command if the commit of the last time this"
        "command was run is an ancestor of this commit",
    )
    to_check_group.add_argument(
        "-g",
        "--glob_to_check",
        action="append",
        help="(re)run the command if the files matching the glob have"
        "modified contents since the last time it was run.",
    )

    parser.add_argument("-s", "--stamp_file", required=True)
    parser.add_argument("-c", "--command_to_run", required=True)
    args = parser.parse_args()

    # Note: this won't work for builds not in a git repo.  This also might not work if we're
    # included as a submodule in another git repo?  Not sure
    # https://stackoverflow.com/a/957978
    root = get_output(["git", "rev-parse", "--show-toplevel"])

    stamp_contents = None  # type: T.Optional[str]
    if args.path_to_check:
        need_to_run, stamp_contents = check_path_git_hash(
            args.path_to_check, args.stamp_file, repo_root=root
        )
    elif args.ancestor_commit:
        need_to_run, stamp_contents = check_if_commit_is_ancestor(
            args.ancestor_commit, args.stamp_file
        )
    elif args.glob_to_check:
        need_to_run, stamp_contents = check_globbed_files_digest(
            args.glob_to_check, args.stamp_file
        )
    else:
        assert False, "Should be mutually exclusive"

    if need_to_run:
        try:
            print("Running:\n " + args.command_to_run)
            subprocess.check_call(args.command_to_run.split())
        except subprocess.CalledProcessError:
            print("\nError while executing command:\n {}".format(args.command_to_run))
            sys.exit(1)
        if stamp_contents is not None:
            write_stamp_file(args.stamp_file, stamp_contents)
    else:
        if args.path_to_check:
            print("{} is up to date".format(args.path_to_check))
        else:
            print("Don't need to run {}".format(args.command_to_run))


if __name__ == "__main__":
    main()
