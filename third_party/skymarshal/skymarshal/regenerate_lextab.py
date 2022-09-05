# aclint: py2 py3
"""
NOTE: if running with bazel, should run from the skymarshal directory
"""
from __future__ import absolute_import

import argparse
import imp
import os
import shutil
import subprocess
import sys
import types
import typing as T
from enum import Enum

# try to get build workspace dir if it exists, otherwise just use paths relative to this file
if os.environ.get("SKYMARSHAL_TEST"):
    SKYMARSHAL_DIR = os.path.dirname(__file__)
else:
    SKYMARSHAL_DIR = os.path.dirname(os.path.realpath(__file__))

ATTRS = (
    "_lexliterals",
    "_lexreflags",
    "_lexstateeoff",
    "_lexstateerrorf",
    "_lexstateignore",
    "_lexstateinfo",
    "_lexstatere",
    "_lextokens",
    "_tabversion",
)


class LexerGenerationError(Exception):
    """
    Errors thrown when Lexer regeneration fails
    """


class PyVersion(Enum):

    PY2 = 0
    PY3 = 1


LEXTABS = {
    PyVersion.PY2: os.path.join(SKYMARSHAL_DIR, "lextab_py2.py"),
    PyVersion.PY3: os.path.join(SKYMARSHAL_DIR, "lextab_py3.py"),
}


def get_py_ver():
    # type: () -> PyVersion
    if sys.version_info.major < 3:
        return PyVersion.PY2
    return PyVersion.PY3


class Lextab(object):
    """
    Helper class to compare two instances of the lexstatere
    """

    ATTRS = (
        "_lexliterals",
        "_lexreflags",
        "_lexstateeoff",
        "_lexstateerrorf",
        "_lexstateignore",
        "_lexstateinfo",
        "_lexstatere",
        "_lextokens",
        "_tabversion",
    )

    def __init__(self, lextab_module):
        # type: (types.ModuleType) -> None
        for attrname in self.ATTRS:
            setattr(self, attrname, getattr(lextab_module, attrname))

    def __eq__(self, other):
        # type: (T.Any) -> bool
        if not isinstance(other, Lextab):
            return False

        for attrname in self.ATTRS:
            if attrname == "_lexstatere":
                if not self._compare_lexstatere(other):
                    return False
            if not getattr(self, attrname) == getattr(other, attrname):
                return False

        return True

    def __repr__(self):
        # type: () -> str
        return "Lextab({})".format(
            ", ".join(
                ["{}={}".format(attrname, str(getattr(self, attrname))) for attrname in self.ATTRS]
            )
        )

    def _compare_lexstatere(self, other):
        # type: (Lextab) -> bool
        """
        This attribute requires special handling because the regex is constructed in a
        non-deterministic fashion, and so the actual content of the regex may be sorted
        on an OR-condition. This means that by splitting on the or symbol | it should be
        safe to verify that the regex set matches properly.

        This logic is required because back to back runs of the regenerate function may yield
        different module outputs.
        """
        self_attr = getattr(self, "_lexstatere")["INITIAL"][0]
        self_re = self_attr[0]

        other_attr = getattr(other, "_lexstatere")["INITIAL"][0]
        other_re = other_attr[0]

        self_token = self_attr[1]
        other_token = other_attr[1]
        if sys.version_info.major < 3:
            self_re_sort = "|".join(sorted(self_re.split("|")))
            other_re_sort = "|".join(sorted(other_re.split("|")))
            return sorted(self_token) == sorted(other_token) and self_re_sort == other_re_sort

        # no need to sort in python 3 because the generation is ordered
        return self_re == other_re and self_token == other_token


def regenerate_lextab(py_ver, write=False):
    # type: (PyVersion, bool) -> None

    tokenizer_path = os.path.join(SKYMARSHAL_DIR, "tokenizer.py")
    generated_path = os.path.join(SKYMARSHAL_DIR, "lextab.py")

    try:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["SKYMARSHAL_REGENERATE_LEXER"] = "1"

        # remove any lextab files before doing a comparison
        if os.path.exists(generated_path):
            os.remove(generated_path)

        subprocess.check_call([sys.executable, tokenizer_path], env=env)

        if not write:
            # if not writing out, do a check to see if the files match
            cached = Lextab(imp.load_source("cached", LEXTABS[py_ver]))
            generated = Lextab(imp.load_source("generated", generated_path))
            if not cached == generated:  # pylint: disable=unneeded-not
                raise LexerGenerationError("Attribute mismatches between generated and cached")
        else:
            # write the file out
            shutil.move(generated_path, LEXTABS[py_ver])

    finally:
        if os.path.exists(generated_path):
            os.remove(generated_path)


def get_parser():
    # type: () -> argparse.ArgumentParser
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="check the lextab and do not write if there is an issue",
    )
    return parser


def main():
    # type: () -> None
    parser = get_parser()
    args = parser.parse_args()

    py_ver = get_py_ver()
    regenerate_lextab(py_ver, not args.check)


if __name__ == "__main__":
    main()
