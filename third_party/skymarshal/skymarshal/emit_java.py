# aclint: py2 py3
# mypy: allow-untyped-defs
from __future__ import absolute_import

import argparse  # pylint: disable=unused-import
import collections
import os
import string
import typing as T
import zipfile
from io import BytesIO

from six.moves import range  # pylint: disable=redefined-builtin
from skymarshal import syntax_tree
from skymarshal.emit_helpers import Code, StructBuilder, render
from skymarshal.language_plugin import SkymarshalLanguage

PrimitiveInfo = collections.namedtuple("PrimitiveInfo", ["storage", "constant", "decode", "encode"])


TYPE_MAP = {
    "byte": PrimitiveInfo("byte", None, "# = ins.readByte();", "outs.writeByte(#);"),
    "int8_t": PrimitiveInfo("byte", "(byte) #", "# = ins.readByte();", "outs.writeByte(#);"),
    "int16_t": PrimitiveInfo("short", "(short) #", "# = ins.readShort();", "outs.writeShort(#);"),
    "int32_t": PrimitiveInfo("int", "#", "# = ins.readInt();", "outs.writeInt(#);"),
    "int64_t": PrimitiveInfo("long", "#L", "# = ins.readLong();", "outs.writeLong(#);"),
    "string": PrimitiveInfo(
        "String",
        None,
        "__strbuf = new char[ins.readInt()-1]; for (int _i = 0; _i < __strbuf.length; _i++) __strbuf[_i] = (char) (ins.readByte()&0xff); ins.readByte(); # = new String(__strbuf);",  # pylint: disable=line-too-long
        "__strbuf = new char[#.length()]; #.getChars(0, #.length(), __strbuf, 0); outs.writeInt(__strbuf.length+1); for (int _i = 0; _i < __strbuf.length; _i++) outs.write(__strbuf[_i]); outs.writeByte(0);",
    ),  # pylint: disable=line-too-long
    "boolean": PrimitiveInfo(
        "boolean", None, "# = ins.readByte()!=0;", "outs.writeByte( # ? 1 : 0);"
    ),
    "float": PrimitiveInfo("float", "#f", "# = ins.readFloat();", "outs.writeFloat(#);"),
    "double": PrimitiveInfo("double", "#", "# = ins.readDouble();", "outs.writeDouble(#);"),
}
# handle unsigned types identically to their signed integer counterparts
TYPE_MAP["uint8_t"] = TYPE_MAP["int8_t"]
TYPE_MAP["uint16_t"] = TYPE_MAP["int16_t"]
TYPE_MAP["uint32_t"] = TYPE_MAP["int32_t"]
TYPE_MAP["uint64_t"] = TYPE_MAP["int64_t"]
# handle protobuf types identically to their appropriate standard integer types
TYPE_MAP["sfixed32"] = TYPE_MAP["int32_t"]
TYPE_MAP["ufixed32"] = TYPE_MAP["uint32_t"]
TYPE_MAP["sfixed64"] = TYPE_MAP["int64_t"]
TYPE_MAP["ufixed64"] = TYPE_MAP["uint64_t"]


def get_pinfo(member):
    return TYPE_MAP.get(member.type_ref.name, None)


def make_accessor(member, obj, get_last_array=False):
    """Create a code string that accesses a variable by name."""
    if not obj:
        accessor = member.name
    else:
        accessor = ".".join([obj, member.name])

    if get_last_array:
        num = member.ndim - 1
    else:
        num = member.ndim
    for i in range(num):
        accessor += "[{}]".format(string.ascii_lowercase[i])

    return accessor


def dim_size_access(dim):
    """Get the size of a single array dimension."""
    if dim.dynamic:
        return "this.{}".format(dim.size_str)
    return dim.size_str


class JavaClass(StructBuilder):
    """Helper class for converting a lcm struct into a java class file."""

    def __init__(self, package, struct, args):
        # TODO(jeff): Implement auto arrays for java classes instead of exposing the legacy form.
        super(JavaClass, self).__init__(package, struct, args)
        new_members = []
        for member in self.members:
            if member.ndim:
                for dim in member.dims:
                    if dim.auto_member:
                        new_members.append(dim.auto_member)
            new_members.append(member)
        self.members = new_members

    @property
    def filename(self):
        return "{}.java".format(os.path.join(self.package.name, self.name))

    @property
    def fullpath(self):
        if self.args.java_path:
            return os.path.join(self.args.java_path, self.filename)
        else:
            return self.filename

    @property
    def full_package_name(self):
        if self.args.package_prefix:
            return self.args.package_prefix + "." + self.package.name
        else:
            return self.package.name

    @property
    def comment(self):
        """The class comment at the top of the defintion"""
        # The old lcm-gen doesn't output java comments.
        return None

    def has_string_members(self):
        for member in self.members:
            if member.type_ref.name == "string":
                return True
        return False

    def complex_subtypes(self):
        for member in self.members:
            if not member.type_ref.is_primitive_type():
                yield member.type_ref

    def fixed_size_array_members(self):
        for member in self.members:
            if isinstance(member, syntax_tree.ArrayMember) and member.is_constant_size():
                yield member

    def declare_member(self, member):
        code = Code()
        pinfo = get_pinfo(member)
        code.start(1, "public ")
        if pinfo:
            code.add(pinfo.storage)
        else:
            code.add(member.type_ref.full_name)
        code.add(" %s", member.name)

        if isinstance(member, syntax_tree.ArrayMember):
            for _ in member.dims:
                code.add("[]")
        code.end(";")

        return code.getvalue().rstrip()

    def preallocate_array_member(self, array_member):
        code = Code()
        code.start(0, "%s = new ", array_member.name)
        pinfo = get_pinfo(array_member)
        if pinfo:
            code.add(pinfo.storage)
        else:
            code.add(array_member.type_ref.full_name)
        for dim in array_member.dims:
            code.add("[%s]", dim.size_str)
        code.end(";")
        return code.getvalue().rstrip()

    def define_constants(self):
        for constant in self.constants:
            pinfo = get_pinfo(constant)
            value = pinfo.constant.replace("#", constant.value_str)
            yield pinfo.storage, constant.name, value

    def encode_member(self, member):
        code = Code()
        pinfo = get_pinfo(member)
        accessor = make_accessor(member, "this")
        self.encode_recursive(code, member, pinfo, accessor, 0)
        code(0, " ")
        return code.getvalue()[:-1]  # remove the last newline to match whitespace

    def encode_recursive(self, code, member, pinfo, accessor, depth):
        # base case: primitive array
        if (depth + 1) == member.ndim and pinfo:
            accessor_array = make_accessor(member, "", get_last_array=True)
            if pinfo.storage == "byte":
                dim = member.dims[depth]
                if dim.dynamic:
                    code(2 + depth, "if (this.%s > 0)", dim.size_str)
                    code(3 + depth, "outs.write(this.%s, 0, %s);", accessor_array, dim.size_str)
                else:
                    code(2 + depth, "outs.write(this.%s, 0, %s);", accessor_array, dim.size_str)
                return

        # base case: generic
        if depth == member.ndim:
            code.start(2 + member.ndim, "")
            if pinfo:
                code.add(pinfo.encode.replace("#", accessor))
            else:
                code.add("{}._encodeRecursive(outs);".format(accessor))
            code.end(" ")
            return

        assert member.ndim > depth, member
        dim = member.dims[depth]
        index_char = string.ascii_lowercase[depth]
        code(
            2 + depth,
            "for (int %s = 0; %s < %s; %s++) {",
            index_char,
            index_char,
            dim_size_access(dim),
            index_char,
        )
        self.encode_recursive(code, member, pinfo, accessor, depth + 1)
        code(2 + depth, "}")

    def copy_member(self, member):
        code = Code()
        pinfo = get_pinfo(member)
        accessor = make_accessor(member, "")

        # allocate an array if necessary
        if member.ndim > 0:
            code.start(2, "outobj.%s = new ", member.name)

            if pinfo:
                code.add(pinfo.storage)
            else:
                code.add(member.type_ref.full_name)

            for dim in member.dims:
                code.add("[(int) %s]", dim.size_str)
            code.end(";")

        self.copy_recursive(code, member, pinfo, accessor, 0)
        code.add(" ")

        return code.getvalue()

    def copy_recursive(self, code, member, pinfo, accessor, depth):
        # base case: primitive array
        if (depth + 1) == member.ndim and pinfo:
            dim = member.dims[depth]

            copy_str = "System.arraycopy(this.{accessor}, 0, outobj.{accessor}, 0, {size});".format(
                accessor=make_accessor(member, "", get_last_array=True), size=dim_size_access(dim)
            )

            if dim.dynamic:
                code(2 + depth, "if (this.%s > 0)", dim.size_str)
                code.start(3 + depth, copy_str)
            else:
                code.start(2 + depth, copy_str)

            # XXX: why dont we have a code.end?
            return

        # base case: generic
        if depth == member.ndim:
            if pinfo:

                code.start(2 + member.ndim, "outobj.%s", member.name)
                for index in string.ascii_lowercase[: member.ndim]:
                    code.add("[%s]", index)

                code.add(" = this.%s", member.name)

                for index in string.ascii_lowercase[: member.ndim]:
                    code.add("[%s]", index)

                code.end(";")

            else:
                code(2 + depth, "outobj.%s = this.%s.copy();", accessor, accessor)

            return

        dim = member.dims[depth]

        index = string.ascii_lowercase[depth]
        code(
            2 + depth,
            "for (int %s = 0; %s < %s; %s++) {",
            index,
            index,
            dim_size_access(dim),
            index,
        )

        self.copy_recursive(code, member, pinfo, accessor, depth + 1)

        code(2 + depth, "}")

    def decode_member(self, member):
        # This is the first call of the recursion. Create necessary helpers
        code = Code()
        pinfo = get_pinfo(member)
        accessor = make_accessor(member, "this")

        # Allocate an array if necessary
        if member.ndim > 0:
            code.start(2, "this.%s = new ", member.name)

            if pinfo:
                code.add(pinfo.storage)
            else:
                code.add(member.type_ref.full_name)

            for dim in member.dims:
                code.add("[(int) %s]", dim.size_str)

            code.end(";")

        self.decode_recursive(code, member, pinfo, accessor, 0)

        # add trailing whitespace to match the old types
        code.add(" ")

        return code.getvalue()

    def decode_recursive(self, code, member, pinfo, accessor, depth):
        # base case: primitive array
        if (depth + 1) == member.ndim and pinfo:

            accessor_array = make_accessor(member, "", get_last_array=True)

            # byte array
            if pinfo.storage == "byte":
                dim = member.dims[depth]
                code.start(
                    2 + depth, "ins.readFully(this.%s, 0, %s);", accessor_array, dim.size_str
                )
                return

        # base case: generic
        if depth == member.ndim:
            code.start(2 + member.ndim, "")
            if pinfo:
                code.add(pinfo.decode.replace("#", accessor))
            else:
                code.add(
                    "%s = %s._decodeRecursiveFactory(ins);", accessor, member.type_ref.full_name
                )
            code.end("")

            return

        dim = member.dims[depth]

        index = string.ascii_lowercase[depth]
        code(
            2 + depth,
            "for (int %s = 0; %s < %s; %s++) {",
            index,
            index,
            dim_size_access(dim),
            index,
        )

        self.decode_recursive(code, member, pinfo, accessor, depth + 1)

        code(2 + depth, "}")


class SkymarshalJava(SkymarshalLanguage):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--java", action="store_true", help="generate bindings for java")
        parser.add_argument("--java-path", help="Location for .java files")
        parser.add_argument(
            "--java-srcjar",
            type=str,
            default=None,
            help="Create a single .srcjar file instead of a directory tree",
        )

    @classmethod
    def create_files(
        cls,
        packages,  # type: T.Iterable[syntax_tree.Package]
        args,  # type: argparse.Namespace
    ):
        # type: (...) -> T.Dict[str, T.Union[str, bytes]]
        """Turn a list of lcm packages into java bindings for each struct.

        @param packages: the list of syntax_tree.Package objects
        @param args: the parsed command-line options for lcmgen

        Returns: a map from filename to java class definition
        """
        if not args.java:
            return {}

        file_map = {}
        for package in packages:
            for struct in package.struct_definitions:
                javaclass = JavaClass(package, struct, args)
                file_map[javaclass.fullpath] = render("lcmtype.java.template", lcmtype=javaclass)
            for enum in package.enum_definitions:
                # TODO(jeff): generate a java enum class instead of an equivalent struct
                javaclass = JavaClass(package, enum.equivalent_struct, args)
                file_map[javaclass.fullpath] = render("lcmtype.java.template", lcmtype=javaclass)

        if args.java_srcjar is not None:
            zip_contents = BytesIO()
            with zipfile.ZipFile(zip_contents, "w") as srcjar:
                for path, contents in file_map.items():
                    # NOTE(eric): We provide a ZipInfo object to be hermetic by not encoding the date
                    srcjar.writestr(zipfile.ZipInfo(path), contents.encode("utf-8"))
            file_map = {args.java_srcjar: zip_contents.getvalue()}

        return file_map
