# mypy: allow-untyped-defs
# aclint: py3

from __future__ import annotations

import argparse  # pylint: disable=unused-import
import typing as T

from skymarshal import syntax_tree  # pylint: disable=unused-import


class SkymarshalLanguage:
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError()

    @classmethod
    def create_files(
        cls,
        packages: T.Iterable[syntax_tree.Package],
        args: argparse.Namespace,
    ) -> T.Dict[str, T.Union[str, bytes]]:
        raise NotImplementedError()
