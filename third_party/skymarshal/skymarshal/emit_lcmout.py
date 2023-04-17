# aclint: py2 py3
"""
Generate output files containing just the definition for single LCM types.
This is used by get_type_for_str, used most notably in json_lcm.
"""
from __future__ import absolute_import

import argparse  # pylint: disable=unused-import
import os
import typing as T

from skymarshal import syntax_tree  # pylint: disable=unused-import
from skymarshal.language_plugin import SkymarshalLanguage


class SkymarshalLCMOut(SkymarshalLanguage):
    @classmethod
    def add_args(cls, parser):
        # type: (argparse.ArgumentParser) -> None
        parser.add_argument(
            "--lcmout", action="store_true", help="output sources for each LCM type"
        )
        parser.add_argument("--lcmout-path", default="", help="Location for outputted .lcm sources")

    @classmethod
    def create_files(
        cls,
        packages,  # type: T.Iterable[syntax_tree.Package]
        args,  # type: argparse.Namespace
    ):
        # type: (...) -> T.Dict[str, T.Union[str, bytes]]
        """
        Turn a list of lcm packages into a set of .lcm files, where each file contains the definition
        for exactly one LCM type (struct or enum).
        """
        if not args.lcmout:
            return {}

        file_map = {}  # type: T.Dict[str, T.Union[str, bytes]]
        for package in packages:
            for definition in package.type_definitions.values():
                output_filename = os.path.join(
                    args.lcmout_path, package.name, "{}.lcm".format(definition.name)
                )
                output_contents = "package {};\n\n{}".format(package.name, str(definition))
                file_map[output_filename] = output_contents

        return file_map
