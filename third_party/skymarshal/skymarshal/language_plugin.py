# mypy: allow-untyped-defs

from __future__ import annotations

import argparse
import typing as T

from skymarshal import syntax_tree


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
