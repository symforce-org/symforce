# aclint: py2 py3
# mypy: allow-untyped-defs
from __future__ import absolute_import

import argparse  # pylint: disable=unused-import
import copy
import os
import typing as T
import zipfile
from io import BytesIO

from skymarshal import syntax_tree  # pylint: disable=unused-import
from skymarshal.emit_helpers import Code, EnumBuilder, StructBuilder, render
from skymarshal.language_plugin import SkymarshalLanguage

TypeInfo = T.NamedTuple(
    "TypeInfo",
    [
        ("pytype", str),
        ("initializer", T.Any),
        ("struct_format", T.Optional[str]),
        ("pack_format", T.Optional[str]),
        ("size", T.Optional[int]),
    ],
)

TYPE_MAP = {
    "byte": TypeInfo("int", 0, "B", "B", 1),
    "boolean": TypeInfo("bool", False, "b", "b", 1),
    "int8_t": TypeInfo("int", 0, "b", "b", 1),
    "int16_t": TypeInfo("int", 0, "h", ">h", 2),
    "int32_t": TypeInfo("int", 0, "i", ">i", 4),
    "int64_t": TypeInfo("int", 0, "q", ">q", 8),
    "uint8_t": TypeInfo("int", 0, "B", "B", 1),
    "uint16_t": TypeInfo("int", 0, "H", ">H", 2),
    "uint32_t": TypeInfo("int", 0, "I", ">I", 4),
    "uint64_t": TypeInfo("int", 0, "Q", ">Q", 8),
    "float": TypeInfo("float", 0.0, "f", ">f", 4),
    "double": TypeInfo("float", 0.0, "d", ">d", 8),
    "string": TypeInfo("T.Text", '""', None, None, None),
}
# handle protobuf types identically to their appropriate standard integer types
TYPE_MAP["sfixed32"] = TYPE_MAP["int32_t"]
TYPE_MAP["ufixed32"] = TYPE_MAP["uint32_t"]
TYPE_MAP["sfixed64"] = TYPE_MAP["int64_t"]
TYPE_MAP["ufixed64"] = TYPE_MAP["uint64_t"]


def module_name_for_lcmtype_name(lcmtype_name):
    # NOTE(matt): we add an underscore to the file name to disambiguate between type and module
    # when importing `from lcmtypes.package import name`
    return "_" + lcmtype_name


class PythonClass(StructBuilder):
    """Helper to construct a python class definition."""

    def __init__(self, package, struct, args):
        super(PythonClass, self).__init__(package, struct, args)
        self.cached_unpackers = dict()  # type: T.Dict[str, T.Tuple[int, str]]

    def get_type_info(self, type_ref):
        # type: (syntax_tree.TypeRef) -> TypeInfo
        info = TYPE_MAP.get(type_ref.name)
        if not info:
            # Create a default info for this non-primitive type.
            # NOTE(jeff): both structs and enums support the _default() classmethod now
            constructor = type_ref.name + "._default()"
            info = TypeInfo(type_ref.name, constructor, None, None, 0)
        return info

    def get_imports(self):
        imports = set()
        for member in self.members:
            if not member.type_ref.is_primitive_type():
                package = member.type_ref.package_name
                module = module_name_for_lcmtype_name(member.type_ref.name)
                # If we are importing ourselves, skip
                if package == self.package.name and module == self.module_name:
                    continue
                if self.args.package_prefix:
                    result = "{}.{}.{}".format(self.args.package_prefix, package, module)
                else:
                    result = "{}.{}".format(package, module)
                    # The {}.{} form worked for eigen_lcm packages until we removed subpackages from
                    # all the __init__.pys. It's now necessary to be explicit:
                    if package == "eigen_lcm":
                        result = "lcmtypes." + result

                imports.add((result, result.split(".")[-1][1:]))
        return sorted(imports)

    def member_initializers(self):
        for member in self.members:
            kwarg_default = self.member_kwarg_default(member)
            initializer = self.member_initializer(member, 0)
            type_hint = self.type_hint(member)
            yield member, kwarg_default, initializer, type_hint

    def member_kwarg_default(self, member):
        if member.ndim != 0:
            return None
        type_info = TYPE_MAP.get(member.type_ref.name)
        if type_info:
            return type_info.initializer
        return None

    def member_initializer(self, member, depth):
        if depth == member.ndim:
            return self.get_type_info(member.type_ref).initializer

        if depth == member.ndim - 1 and member.type_ref.name == "byte":
            # Arrays of bytes get treated as strings, so that they can be more
            # efficiently packed and unpacked.
            return 'b""'
        dim = member.dims[depth]
        if dim.dynamic:
            return "[]"
        inner = self.member_initializer(member, depth + 1)
        return "[ {} for dim{} in range({}) ]".format(inner, depth, dim.size_str)

    def type_hint(self, member, depth=0):
        # type: (syntax_tree.Member, int) -> str
        """
        Returns a mypy type hint for the provided member
        """
        if depth == member.ndim:
            return self.get_type_info(member.type_ref).pytype

        if depth == member.ndim - 1 and member.type_ref.name == "byte":
            return "bytes"

        assert isinstance(member, syntax_tree.ArrayMember)

        array_type = "T.List"
        if (
            depth == member.ndim - 1
            and member.type_ref.is_numeric_type()
            and member.dims[depth].dynamic
        ):
            # Python has efficient operations for unpacking into tuples,
            # and we use them for dynamic-length numeric lists, so those cases
            # can be either tuples or lists.
            array_type = "T.Sequence"

        return "{}[{}]".format(array_type, self.type_hint(member, depth + 1))

    def encode_member(self, code, member, accessor, indent):
        if member.type_ref.is_non_string_primitive_type():
            info = self.get_type_info(member.type_ref)
            assert info.pack_format is not None
            packer = self.get_or_create_pack_struct(info.pack_format)
            code(indent, "buf.write(%s.pack(%s))", packer, accessor)
        elif member.type_ref.name == "string":
            code(indent, "__%s_encoded = %s.encode('utf-8')", member.name, accessor)
            packer = self.get_or_create_pack_struct(">I")
            code(indent, "buf.write(%s.pack(len(__%s_encoded)+1))", packer, member.name)
            code(indent, "buf.write(__%s_encoded)", member.name)
            code(indent, 'buf.write(b"\\0")')
        else:
            gpf = "_get_packed_fingerprint"
            ghr = "_get_hash_recursive"
            name = member.type_ref.name
            # pylcm does not implement _get_packed_fingerprint
            # so use _get_hash_recursive instead in this case
            code(indent, "if hasattr(%s, '%s'):", accessor, gpf)
            code(indent + 1, "assert %s.%s() == %s.%s()", accessor, gpf, name, gpf)
            code(indent, "else:")
            code(indent + 1, "assert %s.%s([]) == %s.%s([])", accessor, ghr, name, ghr)
            code(indent, "%s._encode_one(buf)", accessor)

    def decode_list(self, code, member, accessor, indent, dim, is_first):
        suffix = ""
        if not is_first:
            suffix = ")"

        info = self.get_type_info(member.type_ref)

        if member.type_ref.name == "byte":
            if not dim.dynamic:
                length = dim.size_str
            elif dim.auto_member:
                length = "v_{}".format(dim.size_str)
                self.decode_member(code, dim.auto_member, length + " = ", indent, "")
            else:
                length = "self.{}".format(dim.size_str)
            code(indent, "%sbuf.read(%s)%s", accessor, length, suffix)
        elif member.type_ref.name == "boolean":
            if not dim.dynamic:
                assert dim.size_str is not None
                assert info.size is not None
                assert info.struct_format is not None
                length = int(dim.size_str) * info.size
                unpacker = self.get_or_create_pack_struct(
                    ">%s%c" % (dim.size_str, info.struct_format)
                )
                code(
                    indent,
                    "%slist(map(bool, %s.unpack(buf.read(%d))))%s",
                    accessor,
                    unpacker,
                    length,
                    suffix,
                )
            else:
                if dim.auto_member:
                    length = "v_{}".format(dim.size_str)
                    self.decode_member(code, dim.auto_member, length + " = ", indent, "")
                else:
                    length = "self.{}".format(dim.size_str)
                code(
                    indent,
                    "%slist(map(bool, struct.unpack('>%%d%c' %% %s, buf.read(%s))))%s",
                    accessor,
                    info.struct_format,
                    length,
                    length,
                    suffix,
                )
        elif member.type_ref.is_numeric_type():
            if not dim.dynamic:
                assert dim.size_str is not None
                assert info.size is not None
                assert info.struct_format is not None
                length = int(dim.size_str) * info.size
                unpacker = self.get_or_create_pack_struct(
                    ">%s%c" % (dim.size_str, info.struct_format)
                )
                code(
                    indent,
                    "%slist(%s.unpack(buf.read(%d)))%s",
                    accessor,
                    unpacker,
                    length,
                    suffix,
                )
            else:
                if dim.auto_member:
                    length = "v_{}".format(dim.size_str)
                    self.decode_member(code, dim.auto_member, length + " = ", indent, "")
                else:
                    length = "self.{}".format(dim.size_str)
                assert info.size is not None
                size_mult = (" * %d" % info.size) if info.size > 1 else ""
                code(
                    indent,
                    "%sstruct.unpack('>%%d%c' %% %s, buf.read(%s%s))%s",
                    accessor,
                    info.struct_format,
                    length,
                    length,
                    size_mult,
                    suffix,
                )
        else:
            assert 0

    def pack_members(self, code, members, indent):
        """encode multiple members with a single call to struct.pack()."""
        if not members:
            return
        format_str = ">"
        for member in members:
            info = self.get_type_info(member.type_ref)
            assert info.struct_format is not None
            format_str += info.struct_format
        packer = self.get_or_create_pack_struct(format_str)
        code.start(indent, "buf.write({}.pack(".format(packer))
        code.add(", ".join("self." + member.name for member in members))
        code.end("))")

        # Clear the list
        members[:] = []

    def encode_members(self, indent):
        if not self.members:
            return u"    " * indent + u"pass"

        # Optimization: group adjacent non-string primitives together.
        # This allows us to encode them with a single call to struct.pack(...)
        grouping = []

        code = Code()
        for member in self.members:
            if member.ndim == 0:
                if member.type_ref.is_non_string_primitive_type():
                    grouping.append(member)
                else:
                    self.pack_members(code, grouping, indent)
                    accessor = "self." + member.name
                    self.encode_member(code, member, accessor, indent)
            else:
                # Write out any existing simple members.
                self.pack_members(code, grouping, indent)
                accessor = "self." + member.name

                i = 0
                for dim in member.dims[:-1]:
                    accessor += "[i{}]".format(i)
                    if dim.dynamic:
                        code(indent + i, "for i%d in range(self.%s):", i, dim.size_str)
                    else:
                        code(indent + i, "for i%d in range(%s):", i, dim.size_str)
                    i += 1

                last_dim = member.dims[-1]
                if member.type_ref.is_non_string_primitive_type():
                    self.encode_list(code, member, accessor, indent + i, last_dim)
                else:
                    if last_dim.auto_member:
                        length = "v_" + last_dim.size_str
                        code(indent + i, "%s = len(self.%s)", length, member.name)
                        self.encode_member(code, last_dim.auto_member, length, indent + i)
                        code(indent + i, "for i%d in range(v_%s):", i, last_dim.size_str)
                    elif last_dim.dynamic:
                        code(indent + i, "for i%d in range(self.%s):", i, last_dim.size_str)
                    else:
                        code(indent + i, "for i%d in range(%s):", i, last_dim.size_str)
                    accessor += "[i{}]".format(i)
                    self.encode_member(code, member, accessor, indent + i + 1)

        self.pack_members(code, grouping, indent)
        return code.getvalue().rstrip()

    def encode_list(self, code, member, accessor, indent, dim):
        if member.type_ref.name == "byte":
            if dim.auto_member:
                length = "v_" + dim.size_str
                code(indent, "%s = len(%s)", length, accessor)
                self.encode_member(code, dim.auto_member, length, indent)
            elif dim.dynamic:
                length = "self." + dim.size_str
            else:
                length = dim.size_str
            code(indent, "buf.write(bytearray(%s[:%s]))", accessor, length)
        elif member.type_ref.is_numeric_type() or member.type_ref.name == "boolean":
            info = self.get_type_info(member.type_ref)
            if not dim.dynamic:
                assert info.struct_format is not None
                packer = self.get_or_create_pack_struct(
                    ">%s%c" % (dim.size_str, info.struct_format)
                )
                code(
                    indent,
                    "buf.write(%s.pack(*%s[:%s]))",
                    packer,
                    accessor,
                    dim.size_str,
                )
            else:
                if dim.auto_member:
                    length = "v_" + dim.size_str
                    code(indent, "%s = len(%s)", length, accessor)
                    self.encode_member(code, dim.auto_member, length, indent)
                else:
                    length = "self." + dim.size_str
                code(
                    indent,
                    "buf.write(struct.pack('>%%d%c' %% %s, *%s[:%s]))",
                    info.struct_format,
                    length,
                    accessor,
                    length,
                )
        else:
            assert 0

    def decode_member(self, code, member, accessor, indent, suffix):
        if member.type_ref.name == "boolean":
            unpacker = self.get_or_create_pack_struct("b")
            code(indent, "%sbool(%s.unpack(buf.read(1))[0])%s", accessor, unpacker, suffix)
        elif member.type_ref.is_non_string_primitive_type():
            info = self.get_type_info(member.type_ref)
            assert info.pack_format is not None
            unpacker = self.get_or_create_pack_struct(info.pack_format)
            code(
                indent,
                "%s%s.unpack(buf.read(%s))[0]%s",
                accessor,
                unpacker,
                info.size,
                suffix,
            )
        elif member.type_ref.name == "string":
            unpacker = self.get_or_create_pack_struct(">I")
            # TODO: Consider pre-adding a string unpacker (>I) to all classes
            code(indent, "__%s_len = %s.unpack(buf.read(4))[0]", member.name, unpacker)
            code(
                indent,
                "%sbuf.read(__%s_len)[:-1].decode('utf-8', 'replace')%s",
                accessor,
                member.name,
                suffix,
            )
        else:
            name = member.type_ref.name
            code(indent, "%s%s._decode_one(buf)%s", accessor, name, suffix)

    def unpack_members(self, code, members, indent):
        """decode multiple members with a single call to struct.unpack()."""
        if not members:
            return

        format_str = ">"
        size = 0
        for member in members:
            info = self.get_type_info(member.type_ref)
            assert info.size is not None
            assert info.struct_format is not None
            size += info.size
            format_str += info.struct_format

        unpacker = self.get_or_create_pack_struct(format_str)
        suffix = "[0]" if len(members) == 1 else ""
        code.start(indent, "")
        lhs = ", ".join("self." + member.name for member in members)
        code.add(lhs)
        code.add(" = {}.unpack(".format(unpacker))
        code.end("buf.read(%d))%s", size, suffix)

        # Clear the list
        members[:] = []

    def decode_members(self, indent):
        # pylint: disable=too-many-statements, too-many-branches
        code = Code()

        grouping = []

        for member in self.members:

            if member.ndim == 0:
                if (
                    member.type_ref.is_non_string_primitive_type()
                    and member.type_ref.name != "boolean"
                ):
                    grouping.append(member)
                else:
                    self.unpack_members(code, grouping, indent)
                    accessor = "self." + member.name + " = "
                    self.decode_member(code, member, accessor, indent, "")
            else:
                self.unpack_members(code, grouping, indent)

                accessor = "self." + member.name

                # iterate through the dimensions of the member, building up
                # an accessor string, and emitting for loops
                i = 0
                for dim in member.dims[:-1]:

                    if i == 0:
                        code(indent, "%s = []", accessor)
                    else:
                        code(indent + i, "%s.append([])", accessor)

                    if dim.dynamic:
                        code(indent + i, "for i%d in range(self.%s):", i, dim.size_str)
                    else:
                        code(indent + i, "for i%d in range(%s):", i, dim.size_str)

                    if 0 < i < member.ndim - 1:
                        accessor += "[i{}]".format(i - 1)
                    i += 1

                # last dimension
                last_dim = member.dims[-1]

                if member.type_ref.is_non_string_primitive_type():
                    # member is a primitive non-string type.  Emit code to
                    # decode a full array in one call to struct.unpack
                    if i == 0:
                        accessor += " = "
                    else:
                        accessor += ".append("

                    self.decode_list(code, member, accessor, indent + i, last_dim, i == 0)

                else:
                    # member is either a string type or an inner LCM type.  Each
                    # array element must be decoded individually
                    if i == 0:
                        if last_dim.auto_member:
                            self.decode_member(
                                code,
                                last_dim.auto_member,
                                "v_" + last_dim.size_str + " = ",
                                indent + i,
                                "",
                            )

                        code(indent, "%s = []", accessor)
                    else:
                        code(indent + i, "%s.append ([])", accessor)
                        accessor += "[i{}]".format(i - 1)

                    if last_dim.auto_member:
                        code(indent + i, "for i%d in range(v_%s):", i, last_dim.size_str)
                    elif last_dim.dynamic:
                        code(indent + i, "for i%d in range(self.%s):", i, last_dim.size_str)
                    else:
                        code(indent + i, "for i%d in range(%s):", i, last_dim.size_str)

                    accessor += ".append("
                    self.decode_member(code, member, accessor, indent + i + 1, ")")

        self.unpack_members(code, grouping, indent)
        return code.getvalue().rstrip()

    @property
    def module_name(self):
        return "_" + self.name

    @property
    def filename(self):
        return os.path.join(self.package.name, self.module_name) + ".py"

    @property
    def fullpath(self):
        if self.args.python_path:
            return os.path.join(self.args.python_path or "", self.filename)
        else:
            return self.filename

    @property
    def fully_qualified_name(self):
        return self.struct.type_ref.name

    def has_complex_members(self):
        return bool(list(self.complex_members()))

    def complex_members(self):
        for member in self.members:
            if not member.type_ref.is_primitive_type():
                yield member.type_ref.name, member.name

    def get_or_create_pack_struct(self, format_str):
        # type: (str) -> str
        """
        Get a string referring to a class variable capturing a struct.Struct instance for the format_str.

        This is used to avoid paying the cost of parsing the format_str on every decode.
        format_str should be the string passed to `struct.unpack()`. The return value is a string
        referring to the fully qualified cached struct.
        """
        # Adds one where the format string is already known.
        if format_str not in self.cached_unpackers:
            self.cached_unpackers[format_str] = (
                len(self.cached_unpackers),
                'struct.Struct("{}")'.format(format_str),
            )

        index, _ = self.cached_unpackers[format_str]
        return "{}._CACHED_STRUCT_{}".format(self.name, index)

    def cached_structs_block(self, indent):
        # type: (int) -> T.Text
        """Return a code block to define the struct.Struct instances as class members."""
        code = Code()
        for index, struct_str in sorted(self.cached_unpackers.values()):
            code(indent, "_CACHED_STRUCT_{} = {}".format(index, struct_str))

        return code.getvalue().rstrip()

    @property
    def hashable(self):
        return any(notation.name == "#hashable" for notation in self.struct.notations)


class PyEnum(EnumBuilder):
    # TODO: Add unpacker support for enums, which should be pretty easy.
    def decode_value(self):
        info = TYPE_MAP[self.storage_type.name]
        return "struct.unpack('{}', buf.read({}))[0]".format(info.pack_format, info.size)

    def encode_value(self):
        info = TYPE_MAP[self.storage_type.name]
        return "buf.write(struct.pack('{}', self.value))".format(info.pack_format)

    @property
    def module_name(self):
        return "_" + self.name

    @property
    def filename(self):
        return os.path.join(self.package.name, self.module_name) + ".py"

    @property
    def fullpath(self):
        if self.args.python_path:
            return os.path.join(self.args.python_path or "", self.filename)
        else:
            return self.filename


class SkymarshalPython(SkymarshalLanguage):
    @classmethod
    def add_args(cls, parser):
        # type: (argparse.ArgumentParser) -> None
        parser.add_argument("--python", action="store_true", help="generate bindings for python")
        parser.add_argument(
            "--python-package-prefix", default=None, help="override package-prefix for python only"
        )
        parser.add_argument("--python-path", help="Location for .py files")
        # NOTE(jeff): This option is needed when making static lcmtypes as long as they can reference
        # ptree types, because we need to be able to import just one type at a time from a module.
        parser.add_argument(
            "--python-empty-init",
            action="store_true",
            help="Python bindings should have empty __init__.py files",
        )
        # NOTE(eric): This option is needed under Bazel because the lcmtypes directory ends up in
        # several places. See https://www.python.org/dev/peps/pep-0420/#namespace-packages-today
        parser.add_argument(
            "--python-namespace-packages",
            action="store_true",
            help="Python __init__.py files should declare namespace packages",
        )
        parser.add_argument(
            "--python-zip-path", default=None, help="Zip all Python outputs and put them here"
        )

    @classmethod
    def create_module(
        cls,
        package,  # type: syntax_tree.Package
        args,  # type: argparse.Namespace
        file_map,  # type: T.Dict[str, T.Union[str, bytes]]
    ):
        # type: (...) -> T.List[T.Tuple[str, str]]
        module_items = []

        for struct in package.struct_definitions:
            pyclass = PythonClass(package, struct, args)  # type: T.Union[PythonClass, PyEnum]
            file_map[pyclass.fullpath] = render(
                "python_struct_default_wrapper.py.template",
                lcmtype=pyclass,
            )
            module_items.append((pyclass.module_name, pyclass.name))
        for enum in package.enum_definitions:
            pyclass = PyEnum(package, enum, args)
            file_map[pyclass.fullpath] = render(
                "python_enum_default_wrapper.py.template",
                enumtype=pyclass,
            )
            module_items.append((pyclass.module_name, pyclass.name))

        return module_items

    @classmethod
    def create_files(
        cls,
        packages,  # type: T.Iterable[syntax_tree.Package]
        args,  # type: argparse.Namespace
    ):
        # type: (...) -> T.Dict[str, T.Union[str, bytes]]
        """Turn a list of lcm packages into python bindings for each struct.

        @param packages: the list of syntax_tree.Package objects
        @param args: the parsed command-line options for lcmgen

        Returns: a map from filename to python class defintion.
        """
        if not args.python:
            return {}

        if args.python_package_prefix is not None:
            args = copy.copy(args)
            args.package_prefix = args.python_package_prefix

        file_map = {}  # type: T.Dict[str, T.Union[str, bytes]]
        for package in packages:
            # Create a list of (module_name, lcmtype_name) pairs.
            module_items = cls.create_module(package, args, file_map)

            # Create an init for the package.
            init_path = os.path.join(args.python_path or "", package.name, "__init__.py")
            module_items.sort()

            file_map[init_path] = render(
                "python_init.py.template",
                module_items=module_items,
                empty_init=args.python_empty_init,
                namespace_package=args.python_namespace_packages,
            )

        if args.package_prefix:
            # Create an init for the master package
            init_path = os.path.join(args.python_path or "", "__init__.py")
            file_map[init_path] = render(
                "python_init.py.template",
                empty_init=args.python_empty_init,
                namespace_package=args.python_namespace_packages,
            )

        if args.python_zip_path:
            data = BytesIO()
            compressed = zipfile.ZipFile(data, "w")
            for path, contents in file_map.items():
                # NOTE(eric): If just provided the path, Python will include the current
                # time in the generated zip file, which breaks cache keys. Explicitly constructing
                # a ZipInfo instead uses 1/1/1970, so the hash of the zip is consistent.
                compressed.writestr(zipfile.ZipInfo(path), contents)

            compressed.close()

            file_map = {args.python_zip_path: data.getvalue()}

        return file_map
