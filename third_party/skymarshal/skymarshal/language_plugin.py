# aclint: py2 py3
# mypy: allow-untyped-defs
from __future__ import absolute_import

import argparse  # pylint: disable=unused-import
import typing as T

from skymarshal import syntax_tree  # pylint: disable=unused-import


class SkymarshalLanguage(object):
    @classmethod
    def add_args(cls, parser):
        # type: (argparse.ArgumentParser) -> None
        raise NotImplementedError()

    @classmethod
    def create_files(
        cls,
        packages,  # type: T.Iterable[syntax_tree.Package]
        args,  # type: argparse.Namespace
    ):
        # type: (...) -> T.Dict[str, T.Union[str, bytes]]
        raise NotImplementedError()
