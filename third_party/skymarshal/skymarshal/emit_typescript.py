# aclint: py2 py3
# mypy: allow-untyped-defs
"""LCM definition files for TypeScript"""

# TODO(danny): I think this whole module could use a refactor. This code uses a combination of
# Jinja templates and code generation in python to create its outputs, but in my opinion, creating
# a AST pretty-printer might be a cleaner and clearer way of doing code generation. There's not a
# lot of precendent for that, though.

from __future__ import absolute_import

import argparse  # pylint: disable=unused-import
import os
import typing as T

from skymarshal import syntax_tree  # pylint: disable=unused-import
from skymarshal.emit_helpers import BaseBuilder, EnumBuilder, StructBuilder, TemplateRenderer
from skymarshal.language_plugin import SkymarshalLanguage

# pylint: disable=too-many-instance-attributes

# types that don't need to be imported because they're already TS types
TS_DUPLICATE_TYPES = {"string", "boolean"}

# names that can't be used as identifiers for struct members
TS_RESERVED_KEYWORDS = {
    "any",
    "as",
    "boolean",
    "break",
    "case",
    "catch",
    "class",
    "const",
    "constructor",
    "continue",
    "debugger",
    "declare",
    "default",
    "delete",
    "do",
    "else",
    "enum",
    "export",
    "extends",
    "false",
    "finally",
    "for",
    "from",
    "function",
    "get",
    "if",
    "implements",
    "import",
    "in",
    "instanceof",
    "interface",
    "let",
    "module",
    "never",
    "new",
    "null",
    "number",
    "of",
    "package",
    "private",
    "protected",
    "public",
    "require",
    "return",
    "set",
    "static",
    "string",
    "super",
    "switch",
    "symbol",
    "this",
    "throw",
    "true",
    "try",
    "type",
    "typeof",
    "var",
    "void",
    "while",
    "with",
    "yield",
}

# names that we use in our generated code that can't be used for lcm struct members
LCM_TS_RESERVED_IDENTIFIERS = {
    "_hash",
    "_packed_fingerprint",
    "_get_packed_fingerprint",
    "_get_hash_recursive",
    "_reflection_meta",
    "decode",
    "decode_one",
    "_get_encoded_size",
    "encode",
    "encode_one",
}

RESERVED_IDENTIFIERS = TS_RESERVED_KEYWORDS | LCM_TS_RESERVED_IDENTIFIERS

# protobuf types are converted into LCM types before we translate them into TS types
PROTO_TYPE_MAP = {
    "sfixed32": "int32_t",
    "ufixed32": "uint32_t",
    "sfixed64": "int64_t",
    "ufixed64": "uint64_t",
}

# default values for initializing within TS
TYPE_INITALIZER_MAP = {
    "byte": "0",
    "boolean": "false",
    "int8_t": "0",
    "int16_t": "0",
    "int32_t": "0",
    "int64_t": "0",
    "uint8_t": "0",
    "uint16_t": "0",
    "uint32_t": "0",
    "uint64_t": "0",
    "float": "0.0",
    "double": "0.0",
    "string": '""',
}

# Name of the TS constructor for primitive types, used for reflection metadata
TYPE_CONSTRUCTOR_MAP = {
    "byte": "Number",
    "boolean": "Boolean",
    "int8_t": "Number",
    "int16_t": "Number",
    "int32_t": "Number",
    "int64_t": "Number",
    "uint8_t": "Number",
    "uint16_t": "Number",
    "uint32_t": "Number",
    "uint64_t": "Number",
    "float": "Number",
    "double": "Number",
    "string": "String",
}

# Encoded size of each LCM type, used for decoding and encoding logic
TYPE_ENCODED_SIZE_MAP = {
    "byte": 1,
    "boolean": 1,
    "int8_t": 1,
    "int16_t": 2,
    "int32_t": 4,
    "int64_t": 8,
    "uint8_t": 8,
    "uint16_t": 2,
    "uint32_t": 4,
    "uint64_t": 8,
    "float": 4,
    "double": 8,
}


class TsBase(BaseBuilder):
    @property
    def bare_base_path(self):
        # type: () -> str
        """package_name/filename"""
        return os.path.join(self.package.name, self.name)

    @property
    def bare_import_path(self):
        # type: () -> str
        """
        import_dir/package_name/filename

        for lcmtypes to import other lcmtypes
        """
        return os.path.join(self.args.typescript_import_path, self.bare_base_path)

    @property
    def generation_path(self):
        # type: () -> str
        """
        generation_dir/package_name/filename.ts

        for skymarshal to know where files end up
        """
        return os.path.join(self.args.typescript_path, self.bare_base_path + ".ts")

    @property
    def comment(self):
        # type: () -> str
        """
        class comment at top of definition
        taken directly from emit_cpp.py
        """
        num = len(self._comments)
        if num == 0:
            return ""
        elif num == 1:
            return "\n/// " + self._comments[0]
        else:
            lines = ["/**"]
            for comment in self._comments:
                if comment:
                    lines.append(" * " + comment)
                else:
                    lines.append(" *")
            lines.append(" */")
            return "\n" + "\n".join(lines)


class TsEnum(EnumBuilder, TsBase):
    def __init__(self, package, enum, args):
        # type: (syntax_tree.Package, syntax_tree.Enum, T.Any) -> None
        super(TsEnum, self).__init__(package, enum, args)
        self.storage_type = TsTypeRef(self.storage_type)


class TsInclude(object):
    def __init__(self, member=None, is_primitive=False, prefix=None):
        self.member = member
        self.is_primitive = is_primitive
        self.prefix = prefix

    @property
    def absolute_path(self):
        if self.is_primitive:
            return "{}/types".format(self.prefix)

        return os.path.join(
            self.prefix, self.member.type_ref.package_name, self.member.type_ref.name
        )

    @property
    def directive(self):
        return '{{ {} }} from "{}"'.format(self.member.type_ref.name, self.absolute_path)

    def __hash__(self):
        return hash(self.directive)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return self.directive


def type_ref_to_reflection_dict(type_ref):
    # type: (TsTypeRef) -> str
    """Takes a type_ref and generates a JS dictionary that corresponds to a ReflectionType in TS.
    Does not handle array types, as TsMember has the information about if a member is an array, but
    the TsTypeRef is unaware.

    Args:
        type_ref: of a primitive or a struct

    Returns:
        a string that corresponds to a ReflectionType in TS.
    """
    if type_ref.is_primitive_type():
        return '{ kind: "primitive", type: %s, typeStr: "%s" }' % (
            type_ref.reflection_constructor,
            type_ref.name,
        )
    else:
        return '{ kind: "struct", type: %s, typeStr: "%s" }' % (
            type_ref.reflection_constructor,
            type_ref.name,
        )


class TsMember(object):
    """
    Passthrough type for Member that defines specifically needed extra methods
    """

    def __init__(self, member):
        # type: (syntax_tree.Member) -> None
        self._member = member
        self.type_ref = TsTypeRef(member.type_ref)

    @property
    def name(self):
        """
        Define a safe name for constructor args. There's a type that has the
        name 'var', which is a reserved keyword. Rewrite to _var.
        Checks all typescript keywords
        """
        original_name = self._member.name
        # TODO(danny): add list of TS types
        # prevent field names like 'var' from breaking ts compile
        if original_name in RESERVED_IDENTIFIERS:
            return "_" + original_name
        return original_name

    @property
    def has_auto_member(self):
        return self.ndim == 1 and self.dims[0].auto_member

    @property
    def auto_member_type_ref(self):
        # type: () -> T.Optional[TsTypeRef]
        """
        We need to know storage_size for auto_members, wrap
        """
        if self.has_auto_member:
            return TsTypeRef(self.dims[0].auto_member.type_ref)
        return None

    @property
    def is_byte_array(self):
        # type: () -> bool
        return self.ndim == 1 and self.type_ref.name == "byte"

    @property
    def type_declaration(self):
        # type: () -> str
        """
        Canonical TS definition for a type
        Example: double[][] for a 2d array member of type double
        """
        if self.is_byte_array:
            # special handling for byte arrays of dim == 1
            return "Uint8Array"
        return "{}{}".format(self.type_ref.name, "[]" * self.ndim)

    @property
    def type_reflection_array_dims(self):
        # type: () -> str
        """Reflection metadata generation for array dimensions for ReflectionTypeArray

        Returns:
            JS string that represents an ArrayDim
        """
        if self.has_auto_member:
            return '[{ kind: "auto" }]'

        dim_strings = []
        for dim in self.dims:
            if dim.dynamic:
                dim_strings.append('{ kind: "dynamic", member: "%s" }' % dim.size_str)
            else:
                dim_strings.append('{ kind: "fixed", size: %s }' % dim.size_str)

        return "[{}]".format(", ".join(dim_strings))

    @property
    def type_reflection_dict(self):
        # type: () -> str
        """Reflection metadata generation for this member, corresponds to a ReflectionType

        Returns:
            JS string that represents a ReflectionType
        """
        if self.is_byte_array:
            return '{ kind: "bytes", dims: %s, nDim: 1 }' % (self.type_reflection_array_dims)

        elif self.ndim > 0:
            return '{ kind: "array", dims: %s, nDim: %s, inner: %s }' % (
                self.type_reflection_array_dims,
                self.ndim,
                type_ref_to_reflection_dict(self.type_ref),
            )
        else:
            return type_ref_to_reflection_dict(self.type_ref)

    def __getattr__(self, attr):
        return getattr(self._member, attr)

    def __setattr__(self, attr, val):
        if attr in ("_member", "type_ref"):
            return object.__setattr__(self, attr, val)
        return setattr(self._member, attr, val)

    @property
    def default_initializer(self):
        """
        default zero value for a member
        """
        if self.is_byte_array:
            # special handling for byte arrays of dim == 1
            return "new Uint8Array(0)"
        elif self.ndim == 1 and not self.dims[0].dynamic and self.type_ref.is_primitive_type():
            # special handling for 1D statically sized
            # arrays of primitives to get initialized to
            # their default values
            return "new Array({}).fill({})".format(
                self.dims[0].size_str, TYPE_INITALIZER_MAP[self.type_ref.name]
            )
        elif self.ndim > 0:
            # if array, just make it a zero array
            return "[]"
        elif self.type_ref.is_primitive_type():
            return TYPE_INITALIZER_MAP[self.type_ref.name]
        return "new {}()".format(self.type_ref.name)


class TsTypeRef(object):
    """
    Passthrough type for TypeRef that overrides some behavior
    """

    def __init__(self, type_ref):
        self._type_ref = type_ref

    @property
    def name(self):
        # type: () -> str
        """Map the protobuf types to the standard types"""
        return PROTO_TYPE_MAP.get(self._type_ref.name, self._type_ref.name)

    @property
    def storage_size(self):
        # type: () -> T.Optional[int]
        return TYPE_ENCODED_SIZE_MAP.get(self.name, None)

    @property
    def reflection_constructor(self):
        # type: () -> str
        """Get the constructor function for a type in JS. Checks a primitive map, or just returns
        the struct/enum name. Array types are handled by TsMember, which calls this for the type
        contained in the array.

        Returns:
            identifier of constructor function
        """
        return TYPE_CONSTRUCTOR_MAP.get(self.name, self._type_ref.name)

    def __getattr__(self, attr):
        return getattr(self._type_ref, attr)

    def __setattr__(self, attr, val):
        if attr == "_type_ref":
            return object.__setattr__(self, attr, val)
        return setattr(self._type_ref, attr, val)


class TsStruct(StructBuilder, TsBase):
    def __init__(self, package, struct, args):
        super(TsStruct, self).__init__(package, struct, args)
        self.members = [TsMember(member) for member in self.members]
        self.constants = [TsMember(member) for member in self.constants]

    def complex_members(self):
        for member in self.members:
            if not member.type_ref.is_primitive_type():
                yield member

    def has_complex_members(self):
        return bool(list(self.complex_members()))

    @property
    def include_list(self):
        primitive_includes = set()
        lcm_includes = set()
        for member in self.members + self.constants:
            # add types used by auto members of simple dynamic arrays
            if member.ndim:
                for dim in member.dims:
                    if dim.auto_member:
                        # don't check TS_DUPLICATE_TYPES because auto_members
                        # can't be boolean/string (they are always array sizes)
                        primitive_includes.add(
                            TsInclude(
                                member=dim.auto_member,
                                is_primitive=True,
                                prefix=self.args.typescript_library_path,
                            )
                        )

            if member.type_ref.is_primitive_type():
                # bail if we don't need to import the type
                if member.type_ref.name in TS_DUPLICATE_TYPES:
                    continue
                primitive_includes.add(
                    TsInclude(
                        member=member, is_primitive=True, prefix=self.args.typescript_library_path
                    )
                )
            else:
                # don't import self
                if member.type_ref.name == self.struct.name:
                    continue
                lcm_includes.add(TsInclude(member=member, prefix=self.args.typescript_import_path))

        return sorted(primitive_includes, key=str) + sorted(lcm_includes, key=str)


class SkymarshalTypeScript(SkymarshalLanguage):
    @classmethod
    def add_args(cls, parser):
        # type: (argparse.ArgumentParser) -> None
        parser.add_argument(
            "--typescript", action="store_true", help="generate typescript definitions"
        )
        parser.add_argument(
            "--typescript-path", default="", help="Location of the .ts file hierarchy"
        )
        parser.add_argument(
            "--typescript-import-path",
            default="",
            help="path prefix to use when importing other generated .ts files",
        )
        parser.add_argument(
            "--typescript-library-path",
            default="",
            help="path prefix to use when importing the typescript_lcm package",
        )
        parser.add_argument("--typescript-index-path", default="")

    @classmethod
    def create_files(
        cls,
        packages,  # type: T.Iterable[syntax_tree.Package]
        args,  # type: argparse.Namespace
    ):
        # type: (...) -> T.Dict[str, T.Union[str, bytes]]
        """Turn a list of lcm packages into a bunch of .ts files

        @param packages: the list of syntax_tree.Package objects
        @param args: the parsed command-line options for lcmgen

        Returns: a map from typescript_filename -> lcm ts file contents
        """
        if not args.typescript:
            return {}

        render = TemplateRenderer(os.path.dirname(__file__))
        file_map = {}
        for package in packages:
            for enum in package.enum_definitions:
                enum_type = TsEnum(package, enum, args)
                file_map[enum_type.generation_path] = render(
                    "typescript_enum.ts.template",
                    enum_type=enum_type,
                    typescript_library_path=args.typescript_library_path,
                )
            for struct in package.struct_definitions:
                struct_type = TsStruct(package, struct, args)
                file_map[struct_type.generation_path] = render(
                    "typescript_struct.ts.template",
                    struct_type=struct_type,
                    typescript_library_path=args.typescript_library_path,
                )

        return file_map
