# mypy: allow-untyped-defs
# aclint: py3
"""Generate protocol buffers definition files."""

from __future__ import annotations

import argparse  # typing # pylint: disable=unused-import
import json
import os
import typing as T

from skymarshal import syntax_tree  # pylint: disable=unused-import
from skymarshal.common_util import camelcase_to_snakecase, snakecase_to_camelcase
from skymarshal.emit_helpers import TemplateRenderer
from skymarshal.language_plugin import SkymarshalLanguage
from skymarshal.syntax_tree import ArrayMember, ConstMember

# pylint: disable=too-many-instance-attributes


class PrimitiveType:
    """A primitive type object that supports templating properties"""

    def __init__(self, lcm_decl, lcm_storage, proto_decl, proto_storage, short_int_warning):
        self.lcm_decl = lcm_decl
        self.lcm_storage = lcm_storage
        self.proto_decl = proto_decl
        self.proto_storage = proto_storage
        self.short_int_warning = short_int_warning

    def convert_pb_to_lcm(self, expression):
        if self.short_int_warning:
            return f"static_cast<{self.lcm_storage}>({expression})"
        return expression

    def single_lcm_to_pb(self, field_name, in_expression):
        return f"out->set_{field_name}({in_expression})"

    def add_lcm_to_pb(self, field_name, in_expression):
        return f"out->add_{field_name}({in_expression})"

    @property
    def default_lcm_value(self):
        if self.lcm_decl == "string":
            return '""'
        return 0


PRIMITIVE_MAP = {
    prim.lcm_decl: prim
    for prim in [
        PrimitiveType("boolean", "int8_t", "bool", "bool", False),
        PrimitiveType("byte", "uint8_t", "int32", "int32_t", True),
        PrimitiveType("int8_t", "int8_t", "sint32", "int32_t", True),
        PrimitiveType("int16_t", "int16_t", "sint32", "int32_t", True),
        PrimitiveType("int32_t", "int32_t", "sint32", "int32_t", False),
        PrimitiveType("int64_t", "int64_t", "sint64", "int64_t", False),
        PrimitiveType("string", "std::string", "string", "std::string", False),
        PrimitiveType("float", "float", "float", "float", False),
        PrimitiveType("double", "double", "double", "double", False),
        PrimitiveType("uint8_t", "uint8_t", "uint32", "uint32_t", True),
        PrimitiveType("uint16_t", "uint16_t", "uint32", "uint32_t", True),
        PrimitiveType("uint32_t", "uint32_t", "uint32", "uint32_t", False),
        PrimitiveType("uint64_t", "uint64_t", "uint64", "uint64_t", False),
        PrimitiveType("sfixed32", "int32_t", "sfixed32", "int32_t", False),
        PrimitiveType("sfixed64", "int64_t", "sfixed64", "int64_t", False),
        PrimitiveType("ufixed32", "uint32_t", "fixed32", "uint32_t", False),
        PrimitiveType("ufixed64", "uint64_t", "fixed64", "uint64_t", False),
    ]
}


class EnumCase:
    """A template-friendly wrapper object for LCM #protobuf Enum Cases"""

    def __init__(self, int_value, name):
        self.int_value = int_value
        self.name = name


class EnumType:
    """A template-friendly wrapper object for LCM #protobuf Enums"""

    def __init__(self, package_name, enum, args):
        self.args = args

        # get the filename base
        proto_filename_base = enum.get_notation_property("#protobuf", "filename")
        if proto_filename_base is None:
            if enum.name.endswith("_t"):
                proto_filename_base = camelcase_to_snakecase(enum.name[:-2])
            else:
                proto_filename_base = camelcase_to_snakecase(enum.name)

        # get the type name
        proto_typename = enum.get_notation_property("#protobuf", "typename")
        if proto_typename is None:
            proto_typename = snakecase_to_camelcase(proto_filename_base)

        # See if we should write an `UNKNOWN = 0;`` alias for 0.
        self.add_unknown_enum_alias = enum.get_notation_property(
            "#protobuf", "add_unknown_enum_alias"
        )

        # enumerated cases
        self.cases = [EnumCase(case.int_value, case.name) for case in enum.cases]
        assert self.cases[0].int_value == 0, "Protobuf enums require that the first enum case be 0"
        for case in self.cases:
            if case.int_value < 0 and not enum.get_notation_property(
                "#protobuf", "allow_negative_enums"
            ):
                raise ValueError(
                    "#protobuf enum {}.{} has negative value for {}={}.  Use "
                    "#protobuf{{allow_negative_enums = true}} to suppress this error.".format(
                        package_name,
                        enum.name,
                        case.name,
                        case.int_value,
                    )
                )

        # reserved ids
        self.reserved_ids = ", ".join(str(field_id) for field_id in sorted(enum.reserved_ids))

        # names for templating
        self.definition_name = f"{package_name}.{enum.name}"
        self.proto_package = package_name
        self.proto_java_package = f"com.skydio.pbtypes.{package_name}"
        self.proto_java_outer_classname = "{}Proto".format(
            snakecase_to_camelcase(camelcase_to_snakecase(proto_filename_base))
        )
        self.proto_message_name = proto_typename
        self.proto_enum_name = "Enum"
        self.proto_reference_name = "{}.{}.{}".format(
            package_name, self.proto_message_name, self.proto_enum_name
        )

        self.lcm_cpp_type = f"::{package_name}::{enum.name}"
        self.proto_cpp_container_type = f"::{package_name}::{self.proto_message_name}"
        self.proto_cpp_type = f"{self.proto_cpp_container_type}::{self.proto_enum_name}"

        # filenames for generated converter sources
        self.proto_filename = f"{package_name}/pbtypes/{proto_filename_base}.proto"

        # include paths
        self.proto_import_path = os.path.join(self.args.proto_import_path, self.proto_filename)
        self.protogen_header = f"pbtypes/gen/{package_name}/{proto_filename_base}.pb.h"

    def convert_pb_to_lcm(self, expression):
        return f"{self.lcm_cpp_type}::from_int({expression})"

    def single_lcm_to_pb(self, field_name, in_expression):
        return "out->set_{}(static_cast<{}>({}.int_value()))".format(
            field_name,
            self.proto_cpp_type,
            in_expression,
        )

    def add_lcm_to_pb(self, field_name, in_expression):
        return "out->add_{}(static_cast<{}>({}.int_value()))".format(
            field_name,
            self.proto_cpp_type,
            in_expression,
        )

    @property
    def default_lcm_value(self):
        return f"{self.lcm_cpp_type}{{}}"


class StructField:
    """A template-friendly wrapper object for LCM #protobuf Struct Fields"""

    def __init__(self, parent, member, type_map, args):
        self.args = args
        self.parent = parent
        self.struct = parent.struct
        self.member = member
        self.lcm_name = member.name
        self.proto_name = member.name.lower()
        self.repeated = isinstance(member, ArrayMember)
        self.field_id = member.field_id
        self.type_ref = member.type_ref
        self._type_map = type_map

    @property
    def proto_type_declaration(self):
        if self.type_ref.package_name is None and self.type_ref.name == "byte" and self.repeated:
            return "bytes"
        if self.type_ref.package_name is None and self.type_ref.name in PRIMITIVE_MAP:
            primitive_info = PRIMITIVE_MAP[self.type_ref.name]
            if primitive_info.short_int_warning and not self.struct.get_notation_property(
                "#protobuf", "allow_short_ints"
            ):
                raise TypeError(
                    "Using type {} is not allowed by default.  Use "
                    "#protobuf{{allow_short_ints = true}} to suppress this error.".format(
                        self.type_ref.name,
                    )
                )
            name = primitive_info.proto_decl
        elif self.referenced_type:
            name = self.referenced_type.proto_reference_name
        else:
            raise KeyError(f"Missing referenced type: {self.type_ref}")
        return f"repeated {name}" if self.repeated else name

    def get_type(self):
        if self.type_ref.package_name is None:
            return PRIMITIVE_MAP[self.type_ref.name]
        return self.referenced_type

    @property
    def referenced_type(self):
        full_type_name = f"{self.type_ref.package_name}.{self.type_ref.name}"
        return self._type_map.get(full_type_name)

    @property
    def pb_to_lcm(self):
        try:
            if self.field_id is None:
                array_field = self.parent.array_for_dim(self.lcm_name)
                if array_field.type_ref.name == "byte":
                    # bytes is a string, so call size on the string
                    return f"out.{self.lcm_name} = in.{array_field.proto_name}().size();"
                # normal fields have a <field>_size() method
                return f"out.{self.lcm_name} = in.{array_field.proto_name}_size();"

            field_type = self.get_type()
            if not self.repeated:
                in_expression = field_type.convert_pb_to_lcm(f"in.{self.proto_name}()")
                return f"out.{self.lcm_name} = {in_expression};"

            in_expression = field_type.convert_pb_to_lcm(f"in.{self.proto_name}(i)")
            if self.type_ref.name == "byte":
                # bytes need special logic to convert between string and vector
                in_expression = f"in.{self.proto_name}()"
                return "out.{} = std::vector<uint8_t>({in_expr}.begin(), {in_expr}.end());".format(
                    self.lcm_name, in_expr=in_expression
                )

            dim = self.struct.member_map[self.lcm_name].dims[0]
            var_max_expression = f"in.{self.proto_name}_size()"
            if dim.size_int is None:
                # dynamic array is easy because we pass along the size
                return "for (int i = 0; i < {}; i++) {{\n  out.{}.push_back({});\n}}".format(
                    var_max_expression,
                    self.lcm_name,
                    in_expression,
                )

            # fixed arrays need to be filled with default values if the protobuf isn't long enough
            main_loop = "for (int i = 0; i < {} && i < {}; i++) {{\n  out.{}[i] = {};\n}}".format(
                dim.size_int,
                var_max_expression,
                self.lcm_name,
                in_expression,
            )
            fill_loop = "for (int i = {}; i < {}; i++) {{\n  out.{}[i] = {};\n}}".format(
                var_max_expression,
                dim.size_int,
                self.lcm_name,
                field_type.default_lcm_value,
            )

            if self.type_ref.name == "string":
                # NOTE(jeff): std::array<std::string> will default-initialize to empty string
                return main_loop

            return f"{main_loop}\n{fill_loop}"
        except Exception as e:
            # If there's an AttributeError in this function, jinja will silently "swallow" it and
            # put an empty string in the generated code, which can make it very hard to figure
            # out what's going on. Print the exception here to make debugging easier.
            print(e)
            raise

    @property
    def lcm_to_pb(self):
        if self.field_id is None:
            array_field = self.parent.array_for_dim(self.lcm_name)
            return f"// skip {self.lcm_name} (size of {array_field.lcm_name})"
        field_type = self.get_type()
        if not self.repeated:
            in_expression = f"in.{self.lcm_name}"
            return f"{field_type.single_lcm_to_pb(self.proto_name, in_expression)};"

        dim = self.struct.member_map[self.lcm_name].dims[0]
        if self.type_ref.name == "byte":
            in_expression = f"in.{self.lcm_name}"
            if dim.auto_member:
                return "out->set_{}(std::string({in_expr}.begin(), {in_expr}.end()));".format(
                    self.proto_name,
                    in_expr=in_expression,
                )
            return "out->set_{}(std::string({expr}.begin(), {expr}.begin() + in.{dim}));".format(
                self.proto_name,
                expr=in_expression,
                dim=dim.size_str,
            )

        max_expression = f"in.{dim.size_str}" if dim.size_int is None else dim.size_int
        if dim.auto_member:
            max_expression = f"in.{self.lcm_name}.size()"
        add_statement = field_type.add_lcm_to_pb(self.proto_name, f"in.{self.lcm_name}[i]")
        return f"for (int i = 0; i < {max_expression}; i++) {{\n  {add_statement};\n}}"


class StructType:
    """A template-friendly wrapper object for LCM #protobuf Structs"""

    def __init__(self, package, struct, type_map, args):
        self.args = args
        self.struct = struct
        self.type_map = type_map

        # get the filename base
        proto_filename_base = struct.get_notation_property("#protobuf", "filename")
        if proto_filename_base is None:
            if struct.name.endswith("_t"):
                proto_filename_base = camelcase_to_snakecase(struct.name[:-2])
            else:
                proto_filename_base = camelcase_to_snakecase(struct.name)

        # get the type name
        proto_typename = struct.get_notation_property("#protobuf", "typename")
        if proto_typename is None:
            proto_typename = snakecase_to_camelcase(proto_filename_base)

        # reserved ids
        self.reserved_ids = ", ".join(str(field_id) for field_id in sorted(struct.reserved_ids))

        # enumerated fields
        self.fields = [
            StructField(self, member, type_map, args)
            for member in struct.members
            if not isinstance(member, ConstMember)
        ]
        self.proto_fields = [field for field in self.fields if field.field_id is not None]

        # names for templating
        self.definition_name = f"{package.name}.{struct.name}"
        self.proto_package = package.name
        self.proto_java_package = f"com.skydio.pbtypes.{package.name}"
        self.proto_java_outer_classname = "{}Proto".format(
            snakecase_to_camelcase(camelcase_to_snakecase(proto_filename_base))
        )
        self.proto_message_name = proto_typename
        self.proto_reference_name = f"{package.name}.{self.proto_message_name}"
        self.proto_cpp_type = f"::{package.name}::{self.proto_message_name}"
        self.lcm_cpp_type = f"::{package.name}::{struct.name}"

        # filenames for generated files sources
        self.proto_filename = f"{package.name}/pbtypes/{proto_filename_base}.proto"
        self.protolcm_filename_h = f"{package.name}/{proto_filename_base}.h"
        self.protolcm_filename_cc = f"{package.name}/{proto_filename_base}.cc"

        # include paths
        self.proto_import_path = os.path.join(self.args.proto_import_path, self.proto_filename)
        # TODO(jeff): don't hard-code these include prefixes
        self.lcmgen_header = f"lcmtypes/{package.name}/{struct.name}.hpp"
        self.protogen_header = f"pbtypes/gen/{package.name}/{proto_filename_base}.pb.h"
        self.protolcm_include_path = os.path.join(
            self.args.protolcm_include_path, self.protolcm_filename_h
        )

    @property
    def default_lcm_value(self):
        return f"{self.lcm_cpp_type}{{}}"

    def array_for_dim(self, dim_name):
        return next(
            field
            for field in self.fields
            if field.repeated and field.member.dims[0].size_str == dim_name
        )

    @property
    def referenced_types(self):
        return sorted(
            {
                field.referenced_type
                for field in self.proto_fields
                if field.referenced_type and field.referenced_type != self
            },
            key=lambda x: x.definition_name,
        )

    @property
    def referenced_structs(self):
        return [
            referenced_type
            for referenced_type in self.referenced_types
            if isinstance(referenced_type, StructType)
            or (
                isinstance(referenced_type, ForeignType)
                and referenced_type.protolcm_include_path != ""
            )
        ]

    def convert_pb_to_lcm(self, expression):
        return f"PbToLcm({expression})"

    def single_lcm_to_pb(self, field_name, in_expression):
        return f"LcmToPb({in_expression}, out->mutable_{field_name}())"

    def add_lcm_to_pb(self, field_name, in_expression):
        return f"LcmToPb({in_expression}, out->add_{field_name}())"


ProtoJsonInfo = T.TypedDict(
    "ProtoJsonInfo",
    {
        "reference_name": str,
        "definition_name": str,
        "proto_import_path": str,
        "protolcm_include_path": str,
        "default_lcm_value": str,
        "pb_to_lcm_template": str,
        "single_lcm_to_pb_template": str,
        "add_lcm_to_pb_template": str,
    },
)


class ForeignType:
    def __init__(self, enum_info: ProtoJsonInfo) -> None:
        self.proto_reference_name = enum_info["reference_name"]
        self.pb_to_lcm_template = enum_info["pb_to_lcm_template"]
        self.proto_import_path = enum_info["proto_import_path"]
        self.protolcm_include_path = enum_info["protolcm_include_path"]
        self.default_lcm_value = enum_info["default_lcm_value"]
        self.definition_name = enum_info["definition_name"]

        self._single_lcm_to_pb_template = enum_info["single_lcm_to_pb_template"]
        self._add_lcm_to_pb_template = enum_info["add_lcm_to_pb_template"]

    def convert_pb_to_lcm(self, expression):
        return self.pb_to_lcm_template.format(expression=expression)

    def single_lcm_to_pb(self, field_name, in_expression):
        return self._single_lcm_to_pb_template.format(
            field_name=field_name, in_expression=in_expression
        )

    def add_lcm_to_pb(self, field_name, in_expression):
        return self._add_lcm_to_pb_template.format(
            field_name=field_name, in_expression=in_expression
        )

    @staticmethod
    def get_proto_info(type_map: T.Dict[str, ProtoType]) -> T.Dict[str, ProtoJsonInfo]:
        data = {}
        for type_wrapper in type_map.values():
            data[type_wrapper.definition_name] = ProtoJsonInfo(  # pylint: disable=not-callable
                reference_name=type_wrapper.proto_reference_name,
                definition_name=type_wrapper.definition_name,
                proto_import_path=type_wrapper.proto_import_path,
                default_lcm_value=type_wrapper.default_lcm_value,
                protolcm_include_path=getattr(type_wrapper, "protolcm_include_path", ""),
                pb_to_lcm_template=type_wrapper.convert_pb_to_lcm("{expression}"),
                single_lcm_to_pb_template=type_wrapper.single_lcm_to_pb(
                    "{field_name}", "{in_expression}"
                ),
                add_lcm_to_pb_template=type_wrapper.add_lcm_to_pb(
                    "{field_name}", "{in_expression}"
                ),
            )

        return data


ProtoType = T.Union[StructType, EnumType]
MaybeForeignType = T.Union[ProtoType, ForeignType]


def make_proto_type_map(
    output_packages: T.Iterable[syntax_tree.Package],
    args: argparse.Namespace,
) -> T.Dict[str, ProtoType]:
    """
    Parses the provided packages, outputing all protobuf types that should be generated from them.

    Args:
        output_packages: the list of syntax_tree.Package objects for which to output types
        args: the parsed command-line options for lcmgen
        referenced_packages: list of additional syntax_tree.Package objects directly referenced,
            but which should not have types generated

    Returns:
        Mapping of type name to object
    """

    # All types from output_packages
    type_map: T.Dict[str, ProtoType] = {}

    # All types from output_packages or referenced_packages
    all_referenceable_types: T.Dict[str, MaybeForeignType] = {}
    for path in args.proto_deps_info or []:
        with open(path) as f:
            data = T.cast(T.Dict[str, ProtoJsonInfo], json.load(f))
            for full_name, enum_info in data.items():
                all_referenceable_types[full_name] = ForeignType(enum_info)

    # iterate over all packages we need to parse, adding those from output_packages
    # into both type_map and all_referenceable_types
    for package in output_packages:
        # add all enums
        for enum in package.enum_definitions:
            if any(notation.name == "#protobuf" for notation in enum.notations):
                enum_type = EnumType(package.name, enum, args)
                if enum_type.definition_name in type_map:
                    raise ValueError(
                        f"Two definitions of the same type: {enum_type.definition_name}"
                    )
                all_referenceable_types[enum_type.definition_name] = enum_type
                type_map[enum_type.definition_name] = enum_type
        # add all structs
        for struct in package.struct_definitions:
            if any(notation.name == "#protobuf" for notation in struct.notations):
                struct_type = StructType(package, struct, all_referenceable_types, args)
                if struct_type.definition_name in type_map:
                    raise ValueError(
                        f"Two definitions of the same type: {struct_type.definition_name}"
                    )
                all_referenceable_types[struct_type.definition_name] = struct_type
                type_map[struct_type.definition_name] = struct_type

    return type_map


class SkymarshalProto(SkymarshalLanguage):
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--proto", action="store_true", help="generate protobuf definitions")
        parser.add_argument(
            "--proto-path", default="", help="Location of the .proto file hierarchy"
        )
        parser.add_argument(
            "--proto-import-path",
            default="",
            help="path prefix to use when importing other generated .proto files",
        )
        parser.add_argument(
            "--proto-deps-info",
            nargs="*",
            help="JSON information about dependent files needed by Protobuf",
        )
        parser.add_argument("--proto-info-out", help="output location for json info")

    @classmethod
    def create_files(
        cls, packages: T.Iterable[syntax_tree.Package], args: argparse.Namespace
    ) -> T.Dict[str, T.Union[str, bytes]]:

        """Turn a list of lcm packages into a bunch of .proto files

        Args:
            packages: the list of syntax_tree.Package objects for which to output types
            args: the parsed command-line options for lcmgen
            referenced_packages: list of additional syntax_tree.Package objects directly referenced

        Returns:
            a map from proto_filename to protocol buffers definition
        """
        if not args.proto:
            return {}

        render = TemplateRenderer(os.path.dirname(__file__))
        type_map = make_proto_type_map(packages, args)
        file_map: T.Dict[str, T.Union[str, bytes]] = {}
        for type_wrapper in type_map.values():
            # set paths
            proto_filename = os.path.join(args.proto_path, type_wrapper.proto_filename)
            # render the files
            if isinstance(type_wrapper, EnumType):
                file_map[proto_filename] = render(
                    "proto_enum.proto.template", enum_type=type_wrapper
                )
            elif isinstance(type_wrapper, StructType):
                file_map[proto_filename] = render(
                    "proto_struct.proto.template", struct_type=type_wrapper
                )
            else:
                raise TypeError(f"Unknown wrapper: {type_wrapper}")

        if args.proto_info_out:
            file_map[args.proto_info_out] = json.dumps(
                ForeignType.get_proto_info(type_map), indent=4, sort_keys=True
            )

        return file_map


class SkymarshalProtoLCM(SkymarshalLanguage):
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--protolcm", action="store_true", help="generate protobuf-lcm converters"
        )
        parser.add_argument(
            "--protolcm-path", default="", help="Directory containing <package>/<type>.h files"
        )
        parser.add_argument(
            "--protolcm-include-path",
            default="",
            help="Path prefix to use when including other protolcm converters",
        )

    @classmethod
    def create_files(
        cls,
        packages: T.Iterable[syntax_tree.Package],
        args: argparse.Namespace,
    ) -> T.Dict[str, T.Union[str, bytes]]:
        """Turn a list of lcm packages into a bunch of .h converter files

        skymarshal can be invoked in two ways, the first is invoking it with the full context of the
        entire repo, which will build up a complete type map, and then use it to generate types.
        The other way is to invoke it with a smaller subset of packages (e.g. one), and provide a
        complete list of referenced packages. This allows for incremental outputs to be generated
        without needing to load the type map of the entire repo every time one type changes.

        Args:
            packages: the list of syntax_tree.Package objects for which to output types
            args: the parsed command-line options for lcmgen

        Returns:
            a map from protolcm_filename to type converter implementation
        """
        if not args.protolcm:
            return {}

        render = TemplateRenderer(os.path.dirname(__file__))
        type_map = make_proto_type_map(packages, args)
        file_map = {}
        for type_wrapper in type_map.values():
            # render the files
            if isinstance(type_wrapper, EnumType):
                # enum converters aren't useful enough to generate.
                continue
            elif isinstance(type_wrapper, StructType):
                # set paths
                filename_h = os.path.join(args.protolcm_path, type_wrapper.protolcm_filename_h)
                filename_cc = os.path.join(args.protolcm_path, type_wrapper.protolcm_filename_cc)
                file_map[filename_h] = render("protolcm.h.template", struct_type=type_wrapper)
                file_map[filename_cc] = render("protolcm.cc.template", struct_type=type_wrapper)
            else:
                raise TypeError(f"Unknown wrapper: {type_wrapper}")

        return file_map
