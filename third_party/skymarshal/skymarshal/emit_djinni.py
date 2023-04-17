# aclint: py2 py3
# mypy: allow-untyped-defs
"""Generate djinni enum definitions and conversions."""
from __future__ import absolute_import

import argparse  # pylint: disable=unused-import
import os
import typing as T

from skymarshal import syntax_tree  # pylint: disable=unused-import
from skymarshal import emit_proto
from skymarshal.common_util import snakecase_to_camelcase
from skymarshal.emit_helpers import TemplateRenderer
from skymarshal.language_plugin import SkymarshalLanguage


class EnumCase(object):
    """A template-friendly wrapper object for LCM #djinni Enum Cases"""

    def __init__(self, int_value, name):
        self.int_value = int_value
        # names for templating
        self.definition_name = name
        self.djinni_idl_name = name.lower()
        self.djinni_name = name.upper()
        self.lcm_name = name
        self.proto_name = name


class EnumType(object):
    """A template-friendly wrapper object for LCM #djinni Enums"""

    # pylint: disable=too-many-instance-attributes

    def __init__(self, package_name, enum, args):
        # converted name strings
        user_idl_name = enum.get_notation_property("#djinni", "idl_name")
        if user_idl_name:
            # TODO(jeff): validate name is correct format
            snakecase_name = user_idl_name
        elif enum.name.endswith("_p") or enum.name.endswith("_t"):
            snakecase_name = enum.name[:-2]  # remove suffix
        else:
            snakecase_name = enum.name
        camelcase_name = snakecase_to_camelcase(snakecase_name)

        # enumerated cases
        self.cases = [EnumCase(case.int_value, case.name) for case in enum.cases]
        self.default_case = self.cases[0]

        # names for templating
        self.definition_name = "{}.{}".format(package_name, enum.name)
        self.djinni_idl_name = snakecase_name
        self.djinni_name = camelcase_name
        self.djinni_namespace = args.djinni_module
        self.lcm_name = enum.name
        self.lcm_package = package_name

        # filenames for generated converter sources
        self.filename_h = "{}/converters/{}.h".format(self.djinni_namespace, snakecase_name)
        self.filename_cc = "{}/converters/{}.cc".format(self.djinni_namespace, snakecase_name)

        # header paths to the underlying type definitions
        self.djinni_header = "djinni/{}/{}.hpp".format(self.djinni_namespace, snakecase_name)
        self.lcm_header = "lcmtypes/{}/{}.hpp".format(self.lcm_package, self.lcm_name)

        if enum.get_notation("#protobuf") is None:
            self.is_protobuf = False
            return
        self.proto_type = emit_proto.EnumType(package_name, enum, args)
        self.is_protobuf = True

        # names for templating
        self.proto_typename = self.proto_type.proto_cpp_type
        self.proto_container_typename = self.proto_type.proto_cpp_container_type
        self.proto_header = self.proto_type.protogen_header


class SkymarshalDjinni(SkymarshalLanguage):
    @classmethod
    def add_args(cls, parser):
        # type: (argparse.ArgumentParser) -> None
        parser.add_argument("--djinni", action="store_true", help="generate converters for djinni")
        parser.add_argument("--djinni-idl-path", help="Full path of the .djinni file")
        parser.add_argument("--djinni-path", help="Location of the .cc and .h files")
        parser.add_argument(
            "--djinni-module", default="djinni_lcm", help="Namespace for djinni code"
        )

    @classmethod
    def create_files(
        cls,
        packages,  # type: T.Iterable[syntax_tree.Package]
        args,  # type: argparse.Namespace
    ):
        # type: (...) -> T.Dict[str, T.Union[str, bytes]]
        """
        Turn a list of lcm packages into a djinni definition file and source files for lcm
        converters.

        @param packages: the list of syntax_tree.Package objects
        @param args: the parsed command-line options for lcmgen

        Returns: a map from filename to djinni definition
        """
        if not args.djinni:
            return {}

        render = TemplateRenderer(os.path.dirname(__file__))
        file_map = {}
        enum_types = []
        for package in packages:
            # TODO(jeff): add struct support?
            for enum in package.enum_definitions:
                if any(notation.name == "#djinni" for notation in enum.notations):
                    enum_types.append(EnumType(package.name, enum, args))

        enum_types.sort(key=lambda x: x.djinni_idl_name)

        for enum_type in enum_types:
            idl_file = "{}.djinni".format(args.djinni_module)
            if args.djinni_idl_path:
                idl_file = os.path.join(args.djinni_idl_path, idl_file, enum_type.djinni_idl_name)
                file_map[idl_file] = render("djinni_idl.djinni.template", enum_types=[enum_type])

        for enum_type in enum_types:
            # set paths
            filename_h = enum_type.filename_h
            filename_cc = enum_type.filename_cc
            if args.djinni_path:
                filename_h = os.path.join(args.djinni_path, enum_type.filename_h)
                filename_cc = os.path.join(args.djinni_path, enum_type.filename_cc)
            # render the files
            file_map[filename_h] = render("djinni_converter.h.template", enum_type=enum_type)
            file_map[filename_cc] = render("djinni_converter.cc.template", enum_type=enum_type)

        return file_map
