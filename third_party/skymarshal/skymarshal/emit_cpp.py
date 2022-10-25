# aclint: py2 py3
# mypy: allow-untyped-defs
from __future__ import absolute_import

import argparse  # pylint: disable=unused-import
import copy
import os
import typing as T

import six
from six.moves import range  # pylint: disable=redefined-builtin
from skymarshal import syntax_tree
from skymarshal.emit_helpers import BaseBuilder, Code, EnumBuilder, StructBuilder, render
from skymarshal.language_plugin import SkymarshalLanguage
from skymarshal.syntax_tree import ArrayMember

TYPE_MAP = {
    "string": "std::string",
    "boolean": "int8_t",
    "byte": "uint8_t",
    "sfixed32": "int32_t",
    "sfixed64": "int64_t",
    "ufixed32": "uint32_t",
    "ufixed64": "uint64_t",
}
FUNC_MAP = {
    "sfixed32": "int32_t",
    "sfixed64": "int64_t",
    "ufixed32": "uint32_t",
    "ufixed64": "uint64_t",
}

INTEGER_TYPES = syntax_tree.INTEGER_TYPES + syntax_tree.PROTOBUF_INTEGER_TYPES
INT64_TYPES = ["int64_t", "uint64_t", "ufixed32", "ufixed64"]


def get_array_type(member, mapped_typename):
    std_type_str = "std::array< " if member.is_constant_size() else "std::vector< "
    parts = []
    for _ in member.dims:
        parts.append(std_type_str)
    parts.append(mapped_typename)
    # Reverse dimmension order so that we can write them from the inside out
    for dim in reversed(member.dims):
        if member.is_constant_size():
            parts.append(", " + dim.size_str)
        parts.append(" >")
    return "".join(parts)


def declare_member(member, suffix="", const_ref=False):
    mapped_typename = map_to_cpptype(member.type_ref)
    if isinstance(member, ArrayMember):
        type_str = get_array_type(member, mapped_typename)
    else:
        type_str = mapped_typename

    if const_ref:
        type_str = "const {}&".format(type_str)

    return "{} {}{}".format(type_str, member.name, suffix)


def map_to_cpptype(type_ref):
    """Convert certain types to C++ equivalents, pass through everything else"""
    if type_ref.is_primitive_type():
        return TYPE_MAP.get(type_ref.name, type_ref.name)

    # Use the C++ scope operator instead of a dot
    return "::" + type_ref.full_name.replace(".", "::")


def map_to_functype(type_ref):
    """
    Convert specialized protobuf types to C++ equivalents, pass through everything else.
    This is used for picking from the methods defined in lcm_coretypes.h
    """
    if type_ref.is_primitive_type():
        return FUNC_MAP.get(type_ref.name, type_ref.name)


def dim_size_access(dim, array_member=None):
    """Get the size of a single array dimension"""
    if dim.auto_member:
        if array_member:
            return "this->{}.size()".format(array_member.name)
        return "v_{}".format(dim.size_str)
    if dim.dynamic:
        return "this->{}".format(dim.size_str)
    return dim.size_str


class CppBase(BaseBuilder):
    """Base helper for lcm stucts and enums"""

    @property
    def namespace(self):
        return self.package.name

    @property
    def filename(self):
        return "{}.hpp".format(os.path.join(self.namespace, self.name))

    @property
    def fullpath(self):
        if self.args.cpp_hpath:
            return os.path.join(self.args.cpp_hpath, self.filename)
        else:
            return self.filename

    @property
    def underscored(self):
        return "{}_{}".format(self.namespace, self.name)

    @property
    def comment(self):
        """The class comment at the top of the defintion"""
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

    @property
    def size_t(self):
        # type: () -> str
        """
        The C++ type to use for buffer size variables
        """
        return "__lcm_buffer_size"


class CppEnum(EnumBuilder, CppBase):
    """Helper class for converting an lcm enum into a C++ class file.

    Because there is no source compatibility constraint, this is easy to do entirely in jinja2
    """

    @property
    def string_cast_type(self):
        """If printing a number, cast int8_t to int16_t otherwise it is treated as a character"""
        storage_name = str(self.storage_type)
        return {"int8_t": "int16_t"}.get(storage_name, storage_name)


class CppInclude(object):
    def __init__(self, member=None, std=None, prefix=None):
        assert member or std
        self.member = member
        self.std = std
        self.prefix = prefix

    @property
    def relative_path(self):
        if self.std:
            return self.std
        return "{}.hpp".format(
            os.path.join(self.member.type_ref.package_name, self.member.type_ref.name)
        )

    @property
    def absolute_path(self):
        if not self.prefix:
            return self.relative_path
        return os.path.join(self.prefix, self.relative_path)

    @property
    def directive(self):
        name = self.absolute_path
        if self.std is not None:
            name = "<{}>".format(name)
        else:
            name = '"{}"'.format(name)
        return "#include {}\n".format(name)

    @property
    def package(self):
        return self.member.type_ref.package_name

    def __hash__(self):
        return hash(self.directive)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return self.directive


class CppStruct(StructBuilder, CppBase):
    """Helper class for converting a lcm struct into a C++ class file.

    This uses a combination of handwritten generation and jinja2.
    Ideally more would be in jinja2, but I was copying from lcm/lcmgen/emit_cpp.c
    and I wanted the whitespace to match exactly for testing purposes.
    """

    def include_list(self):
        "A list of includes required to compile this class"
        # include header files for other LCM types
        includes = set()
        for member in self.members:
            if isinstance(member, ArrayMember):
                if member.is_constant_size():
                    includes.add(CppInclude(std="array"))
                else:
                    includes.add(CppInclude(std="vector"))
            if member.type_ref.name == "string":
                includes.add(CppInclude(std="string"))

            include = CppInclude(member=member, prefix=self.args.cpp_include)
            if not member.type_ref.is_primitive_type():
                # This is an lcmtype
                if include.relative_path != self.filename:
                    includes.add(include)
        std_includes = set(inc for inc in includes if inc.std)
        lcm_includes = includes - std_includes
        return sorted(std_includes, key=str) + sorted(lcm_includes, key=str)

    def includes(self):
        "The includes required to compile this class as a formatted string."
        includes = self.include_list()
        if not includes:
            return ""
        else:
            return "".join(include.directive for include in includes)

    def declare_members(self):
        "return a list of type declarations for each member"
        code = Code()
        for member in self.members:
            for comment in member.comments:
                code(2, "// %s", comment)
            declaration = declare_member(member)
            code(2, declaration + ";\n")
        return code.getvalue().rstrip()

    def define_constants(self):
        code = Code()
        for constant in self.constants:
            # emit_comment(f, 2, lc->comment);
            # For integers only, we emit enums instead of static const
            # values because the former can be passed by reference while
            # the latter cannot.
            if constant.type_ref.name in INTEGER_TYPES:
                suffix = "LL" if constant.type_ref.name in INT64_TYPES else ""
                code(
                    2,
                    "enum : %s { %s = %s%s };",
                    map_to_cpptype(constant.type_ref),
                    constant.name,
                    constant.value_str,
                    suffix,
                )
            else:
                mapped_typename = map_to_cpptype(constant.type_ref)
                if self.args.cpp_std not in ("c++98", "c++11"):
                    raise ValueError("Invalid C++ std: {}".format(self.args.cpp_std))

                if self.args.cpp_std == "c++11":
                    code(2, "// If you are getting compiler/linker errors saying things like")
                    code(2, "// undefined reference to pkg::type::kConstant,")
                    code(2, "// that's because this isn't technically correct on C++ <17.")
                    code(2, "// Until we find a better solution, you can probably make it go away")
                    code(2, "// with -O2.")
                    code(
                        2,
                        "static constexpr %-8s %s = %s;",
                        mapped_typename,
                        constant.name,
                        constant.value_str,
                    )
                else:
                    code(
                        2,
                        "// If you're using C++11 and are getting compiler errors saying things"
                        " like",
                    )
                    code(
                        2,
                        "// 'constexpr' needed for in-class initialization of static data"
                        " member",
                    )
                    code(
                        2,
                        "// then re-run lcm-gen with '--cpp-std=c++11' to generate code that is",
                    )
                    code(2, "// compliant with C++11")
                    code(
                        2,
                        "static const %-8s %s = %s;",
                        mapped_typename,
                        constant.name,
                        constant.value_str,
                    )
        return code.getvalue().rstrip()

    def constructor_args(self):
        return ",\n".join(declare_member(member, "_arg", const_ref=True) for member in self.members)

    def initializers(self):
        return ",\n".join("{0}({0}_arg)".format(member.name) for member in self.members)

    def encode_recursive(self, code, member, depth, extra_indent):
        # type: (Code, syntax_tree.Member, int, int) -> None
        indent = extra_indent + 1 + depth
        # primitive array
        if member.ndim == (depth + 1) and member.type_ref.is_non_string_primitive_type():
            dim = member.dims[depth]  # type: ignore[attr-defined]
            code.start(
                indent,
                "tlen = __%s_encode_array(buf, offset + pos, maxlen - pos, &this->%s",
                map_to_functype(member.type_ref),
                member.name,
            )
            for i in range(depth):
                code.add("[a%d]", i)
            code.end("[0], %s);", dim_size_access(dim))

            code(indent, "if(tlen < 0) return tlen; else pos += tlen;")
            # this is the end of the array
            return

        # other, or no arrays
        if depth == member.ndim:
            if member.type_ref.name == "string":
                code.start(indent, "char* __cstr = (char*) this->%s", member.name)
                for i in range(depth):
                    code.add("[a%d]", i)
                code.end(".c_str();")
                code(
                    indent,
                    "tlen = __string_encode_array(buf, offset + pos, maxlen - pos, &__cstr, 1);",
                )
            else:
                code.start(indent, "tlen = this->%s", member.name)
                for i in range(depth):
                    code.add("[a%d]", i)
                code.end("._encodeNoHash(buf, offset + pos, maxlen - pos);")

            code(indent, "if(tlen < 0) return tlen; else pos += tlen;")
            return

        dim = member.dims[depth]  # type: ignore[attr-defined]

        code(
            indent,
            "for (%s a%d = 0; a%d < %s; a%d++) {",
            self.size_t,
            depth,
            depth,
            dim_size_access(dim),
            depth,
        )

        self.encode_recursive(code, member, depth + 1, extra_indent)

        code(indent, "}")

    def encode_members(self):
        code = Code()
        for member in self.members:
            self.encode_member(code, member)
        return code.getvalue().rstrip()

    def encode_member(self, code, member, virtual=False):
        if isinstance(member, ArrayMember):
            last_dim = member.dims[-1]

            if last_dim.auto_member:
                code(
                    1,
                    "%s %s = %s;",
                    map_to_cpptype(last_dim.auto_member.type_ref),
                    dim_size_access(last_dim),
                    dim_size_access(last_dim, member),
                )
                self.encode_member(code, last_dim.auto_member, virtual=True)

            # for non-string primitive types with variable size final
            # dimension, add an optimization to only call the primitive encode
            # functions only if the final dimension size is non-zero.
            if member.type_ref.is_non_string_primitive_type() and last_dim.dynamic:
                code(1, "if(%s > 0) {", dim_size_access(last_dim))
                self.encode_recursive(code, member, 0, 1)
                code(1, "}")
            else:
                self.encode_recursive(code, member, 0, 0)

        else:
            if member.type_ref.is_primitive_type():
                if member.type_ref.name == "string":
                    code(1, "char* %s_cstr = (char*) this->%s.c_str();", member.name, member.name)
                    code(
                        1,
                        "tlen = __string_encode_array(buf, offset + pos, maxlen - pos,"
                        " &%s_cstr, 1);",
                        member.name,
                    )
                else:
                    size_str = ("v_%s" if virtual else "this->%s") % member.name
                    code(
                        1,
                        "tlen = __%s_encode_array(buf, offset + pos, maxlen - pos," " &%s, 1);",
                        map_to_functype(member.type_ref),
                        size_str,
                    )
                code(1, "if(tlen < 0) return tlen; else pos += tlen;")
            else:
                self.encode_recursive(code, member, 0, 0)
        code(0, "")

    def has_complex_members(self):
        for member in self.members:
            if not member.type_ref.is_primitive_type():
                return True
        return False

    def compute_hash(self):
        hash_calls = []
        for member in self.members:
            if not member.type_ref.is_primitive_type():
                scoped_name = map_to_cpptype(member.type_ref)
                hash_calls.append("{}::_computeHash(&cp)".format(scoped_name))
        return " +\n         ".join(hash_calls)

    def encoded_size(self):
        if len(self.members) == 0:
            return "    return 0;"

        code = Code()
        code(1, "{} enc_size = 0;".format(self.size_t))
        for member in self.members:
            self.encoded_size_member(code, member)

        code(1, "return enc_size;")
        return code.getvalue().rstrip()

    def encoded_size_member(self, code, member):
        if member.ndim > 0:
            for dim in member.dims:
                if dim.auto_member:
                    self.encoded_size_member(code, dim.auto_member)

        if member.type_ref.is_non_string_primitive_type():
            code.start(1, "enc_size += ")

            if member.ndim > 0:
                for i in range(member.ndim - 1):
                    dim = member.dims[i]
                    code.add("%s * ", dim_size_access(dim))
                dim = member.dims[-1]
                code.end(
                    "__%s_encoded_array_size(NULL, %s);",
                    map_to_functype(member.type_ref),
                    dim_size_access(dim, member),
                )
            else:
                code.end("__%s_encoded_array_size(NULL, 1);", map_to_functype(member.type_ref))
        else:
            for i in range(member.ndim):
                dim = member.dims[i]
                code(
                    1 + i,
                    "for (%s a%d = 0; a%d < %s; a%d++) {",
                    self.size_t,
                    i,
                    i,
                    dim_size_access(dim, member),
                    i,
                )
            code.start(member.ndim + 1, "enc_size += this->%s", member.name)
            for i in range(member.ndim):
                code.add("[a%d]", i)
            if member.type_ref.name == "string":
                code.end(".size() + 4 + 1;")
            else:
                code.end("._getEncodedSizeNoHash();")
            for i in range(member.ndim, 0, -1):
                code(i, "}")

    def decode_recursive(self, code, member, depth):
        # primitive array
        if depth + 1 == member.ndim and member.type_ref.is_non_string_primitive_type():
            dim = member.dims[depth]

            decode_indent = 1 + depth

            if not member.is_constant_size():
                code.start(1 + depth, "this->%s", member.name)
                for i in range(depth):
                    code.add("[a%d]", i)
                code.end(".resize(%s);", dim_size_access(dim))

                code(1 + depth, "if(%s) {", dim_size_access(dim))

                decode_indent += 1

            code.start(
                decode_indent,
                "tlen = __%s_decode_array(buf, offset + pos, maxlen - pos, &this->%s",
                map_to_functype(member.type_ref),
                member.name,
            )

            for i in range(depth):
                code.add("[a%d]", i)

            code.end("[0], %s);", dim_size_access(dim))
            code(decode_indent, "if(tlen < 0) return tlen; else pos += tlen;")

            if not member.is_constant_size():
                code(1 + depth, "}")

        elif depth == member.ndim:
            if member.type_ref.name == "string":
                code(1 + depth, "int32_t __elem_len;")
                code(
                    1 + depth,
                    "tlen = __int32_t_decode_array(buf, offset + pos, maxlen - pos,"
                    " &__elem_len, 1);",
                )
                code(1 + depth, "if(tlen < 0) return tlen; else pos += tlen;")
                code(1 + depth, "if(__elem_len > maxlen - pos) return -1;")
                code.start(1 + depth, "this->%s", member.name)
                for i in range(depth):
                    code.add("[a%d]", i)
                code.end(".assign(((const char*)buf) + offset + pos, __elem_len -  1);")
                code(1 + depth, "pos += __elem_len;")
            else:
                code.start(1 + depth, "tlen = this->%s", member.name)
                for i in range(depth):
                    code.add("[a%d]", i)
                code.end("._decodeNoHash(buf, offset + pos, maxlen - pos);")
                code(1 + depth, "if(tlen < 0) return tlen; else pos += tlen;")
        else:
            dim = member.dims[depth]
            if not member.is_constant_size():
                code.start(1 + depth, "this->%s", member.name)
                for i in range(depth):
                    code.add("[a%d]", i)
                code.end(".resize(%s);", dim_size_access(dim))

            code(
                1 + depth,
                "for (%s a%d = 0; a%d < %s; a%d++) {",
                self.size_t,
                depth,
                depth,
                dim_size_access(dim),
                depth,
            )

            self.decode_recursive(code, member, depth + 1)

            code(1 + depth, "}")

    def decode_members(self):
        code = Code()
        for member in self.members:
            self.decode_member(code, member)
        return code.getvalue().rstrip()

    def decode_member(self, code, member, virtual=False):
        if member.ndim > 0:
            for dim in member.dims:
                if dim.auto_member:
                    code(
                        1, "%s %s;", map_to_cpptype(dim.auto_member.type_ref), dim_size_access(dim)
                    )
                    self.decode_member(code, dim.auto_member, virtual=True)

        if 0 == member.ndim and member.type_ref.is_primitive_type():
            if member.type_ref.name == "string":
                code(1, "int32_t __%s_len__;", member.name)
                code(
                    1,
                    "tlen = __int32_t_decode_array(buf, offset + pos, maxlen - pos,"
                    " &__%s_len__, 1);",
                    member.name,
                )
                code(1, "if(tlen < 0) return tlen; else pos += tlen;")
                code(1, "if(__%s_len__ > maxlen - pos) return -1;", member.name)
                code(
                    1,
                    "this->%s.assign(((const char*)buf) + offset + pos, __%s_len__ - 1);",
                    member.name,
                    member.name,
                )
                code(1, "pos += __%s_len__;", member.name)
            else:
                size_str = ("v_%s" if virtual else "this->%s") % member.name
                code(
                    1,
                    "tlen = __%s_decode_array(buf, offset + pos, maxlen - pos, &%s," " 1);",
                    map_to_functype(member.type_ref),
                    size_str,
                )
                code(1, "if(tlen < 0) return tlen; else pos += tlen;")

        else:
            self.decode_recursive(code, member, 0)
        code(0, "")

    @property
    def cpp_display_everywhere(self):
        return any(notation.name == "#cpp_display_everywhere" for notation in self.struct.notations)

    @property
    def cpp_no_display(self):
        return any(notation.name == "#cpp_no_display" for notation in self.struct.notations)


class SkymarshalCpp(SkymarshalLanguage):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--cpp", action="store_true", help="generate bindings for cpp")
        parser.add_argument(
            "--cpp-package-prefix", default=None, help="override package-prefix for cpp only"
        )
        parser.add_argument("--cpp-std", default="c++11", help="C++ standard(c++98, c++11)")
        parser.add_argument("--cpp-hpath", help="Location for .hpp files")
        parser.add_argument("--cpp-include", help="Generated #include lines reference this folder")
        parser.add_argument(
            "--cpp-forward",
            action="store_true",
            help="Generates forward-declarations in {module}_fwd.hpp",
        )
        parser.add_argument(
            "--cpp-aggregate", action="store_true", help="Generates aggregate {module}.hpp"
        )
        parser.add_argument(
            "--cpp-no-operators",
            action="store_true",  # TODO(jeff): use for enums?
            help="Exclude operators and member constructor for structs",
        )

    @classmethod
    def create_files(
        cls,
        packages,  # type: T.Iterable[syntax_tree.Package]
        args,  # type: argparse.Namespace
    ):
        # type: (...) -> T.Dict[str, T.Union[str, bytes]]
        """Turn a list of lcm packages into C++ bindings for each struct.

        @param packages: the list of syntax_tree.Package objects
        @param args: the parsed command-line options for lcmgen

        Returns: a map from filename to header file content.
        """
        if not args.cpp:
            return {}

        if args.cpp_package_prefix is not None:
            args = copy.copy(args)
            args.package_prefix = args.cpp_package_prefix

        file_map = {}
        for package in packages:
            for struct in package.struct_definitions:
                cppclass = CppStruct(package, struct, args)  # type: T.Union[CppStruct, CppEnum]
                file_map[cppclass.fullpath] = render(
                    "lcmtype.hpp.template",
                    lcmtype=cppclass,
                    is_array_member=lambda member: isinstance(member, ArrayMember),
                    array_type_str=lambda member: get_array_type(
                        member, map_to_cpptype(member.type_ref)
                    ),
                    size_t=cppclass.size_t,
                )
            for enum in package.enum_definitions:
                cppclass = CppEnum(package, enum, args)
                file_map[cppclass.fullpath] = render(
                    "enumtype.hpp.template", enumtype=cppclass, size_t=cppclass.size_t
                )

            type_names = sorted(
                type_definition.name for type_definition in six.itervalues(package.type_definitions)
            )

            if args.cpp_aggregate:
                path = os.path.join(args.cpp_hpath, package.name + ".hpp")
                file_map[path] = render(
                    "aggregate.hpp.template", package_name=package.name, type_names=type_names
                )

            if args.cpp_forward:
                path = os.path.join(args.cpp_hpath, package.name + "_fwd.hpp")
                file_map[path] = render(
                    "forward.hpp.template", package_name=package.name, type_names=type_names
                )

        return file_map
