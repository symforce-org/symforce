# aclint: py2 py3
# mypy: allow-untyped-defs
from __future__ import absolute_import, print_function

import typing as T

from numpy import int64
from six import iteritems, next  # pylint: disable=redefined-builtin

INTEGER_TYPES = (
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
)
PROTOBUF_INTEGER_TYPES = ("sfixed32", "ufixed32", "sfixed64", "ufixed64")
FLOAT_TYPES = ("float", "double")
NUMERIC_TYPES = INTEGER_TYPES + PROTOBUF_INTEGER_TYPES + FLOAT_TYPES

# These are the types allowed to be used in constants.
CONST_TYPES = INTEGER_TYPES + FLOAT_TYPES

# NOTE(eric): This should be synchronized with the definitions in tools/gazelle/lcm/lcmparser.go
PRIMITIVE_TYPES = NUMERIC_TYPES + ("boolean", "byte", "string")

CONST_TYPE_MAP = {}  # type: T.Dict[str, T.Any]
for integer_type in INTEGER_TYPES:
    CONST_TYPE_MAP[integer_type] = lambda num: int(num, base=0)  # automatically detect hex or dec.

for float_type in FLOAT_TYPES:
    CONST_TYPE_MAP[float_type] = float


class Hash(object):
    def __init__(self):
        self.val = int64(0x12345678)
        # TODO(matt): is it possible to remove the int64 dependency?

    def update(self, byte):
        """Make the hash dependent on the value of the given character.
        The order that hash_update is called in IS important."""
        self.val = ((self.val << 8) ^ (self.val >> 55)) + byte

    def update_string(self, string):
        "Make the hash dependent on each character in a string."
        self.update(len(string))

        for char in string:
            # NOTE(matt): This restricts the names to be ASCII characters
            self.update(ord(char))

    @property
    def int_value(self):
        # convert to uint64-like number
        # NOTE(matt): this is still a python int, and thus has infinite size
        return int(self.val) & 0xFFFFFFFFFFFFFFFF

    def hex_no_padding(self):
        return "0x{0:x}".format(self.int_value)

    def hex_str(self):
        # Convert to hexidecimal with padding
        return "{0:#0{1}x}".format(self.int_value, 18)  # 18 is 16 + 2

    def __repr__(self):
        # Add the L since most languages use that.
        return self.hex_str() + "L"

    def __eq__(self, rhs):
        return self.int_value == rhs


class AstNode(object):
    """Base class in the syntax tree"""

    def __init__(self):
        self.lineno = -1
        self.comments = []


class Package(AstNode):
    """A named container of type definitions"""

    def __init__(self, name, type_definitions):
        # type: (str, T.Sequence[T.Union[Enum, Struct]]) -> None
        super(Package, self).__init__()
        self.name = name
        self.type_definitions = {t.name: t for t in type_definitions}
        for type_definition in self.type_definitions.values():
            type_definition.add_package_name(self.name)

    @property
    def struct_definitions(self):
        return [s for s in self.type_definitions.values() if isinstance(s, Struct)]

    @property
    def enum_definitions(self):
        return [e for e in self.type_definitions.values() if isinstance(e, Enum)]

    def reference_check(self):
        for type_definition in self.type_definitions.values():
            type_definition.reference_check()

    def type_definition_with_name(self, name):
        """Return a struct in the package by name."""
        if name not in self.type_definitions:
            raise NameError(
                "Struct/Enum `{}` not found in package `{}`. Did you forget to run make?".format(
                    name,
                    self.name,
                )
            )
        # should be no duplicates
        return self.type_definitions[name]

    def extend_with_package(self, other_package):
        """Add the given package's structs and enums to this package"""
        assert self.name == other_package.name
        for name, t in iteritems(other_package.type_definitions):
            if name in self.type_definitions:
                raise NameError(
                    "Struct/Enum `{}` duplicated in package `{}`.".format(name, self.name)
                )
            self.type_definitions[name] = t

    def __repr__(self):
        lines = []
        for _, type_definition in iteritems(self.type_definitions):
            lines.extend(repr(type_definition).splitlines())
        children = "\n".join(lines)
        return "package {};\n{}".format(self.name, children)


NotationSpecProperty = T.NamedTuple("NotationSpecProperty", [("name", str), ("type", str)])
NotationSpec = T.NamedTuple(
    "NotationSpec", [("allowed", T.Set[str]), ("properties", T.List[NotationSpecProperty])]
)


class Notation(AstNode):
    NOTATION_SPECS = {
        "#djinni": NotationSpec(
            allowed={"enum"},
            properties=[
                NotationSpecProperty(
                    name="idl_name",
                    type="string",
                ),
            ],
        ),
        "#protobuf": NotationSpec(
            allowed={"enum", "struct"},
            properties=[
                NotationSpecProperty(
                    name="typename",
                    type="string",
                ),
                NotationSpecProperty(
                    name="filename",
                    type="string",
                ),
                NotationSpecProperty(
                    name="allow_short_ints",
                    type="bool",
                ),
                NotationSpecProperty(
                    name="allow_negative_enums",
                    type="bool",
                ),
            ],
        ),
        "#hashable": NotationSpec(
            allowed={"struct"},
            properties=[],
        ),  # enums are always hashable
        # By default, types have a operator<< that prints a human-readable representation. On
        # platforms where the `SKYMARSHAL_PRINTING_ENABLED` macro is NOT defined, the operator is
        # defined, but does not print out a useful representation (it instead prints the constant
        # string `"<FORMATTING DISABLED>"`).
        #
        # Types annotated as `#cpp_no_display` will have no operator<<. This annotation takes
        # precedence over `#cpp_display_everywhere` and `SKYMARSHAL_PRINTING_ENABLED`. This is
        # useful for types that define that operator elsewhere. If you are defining such an
        # operator, make sure to respect `SKYMARSHAL_PRINTING_ENABLED`.
        #
        # A type annotated as `#cpp_display_everywhere` will have a useful human-readable
        # representation regardless of `SKYMARSHAL_PRINTING_ENABLED`.
        "#cpp_display_everywhere": NotationSpec(allowed={"struct"}, properties=[]),
        "#cpp_no_display": NotationSpec(allowed={"struct"}, properties=[]),
    }

    # If allow_unknown_notations is true, then any notation that doesn't match a NOTATION_SPECS
    # prings a warning, and is allowed to have any properties, though those properties are not
    # parsed.
    # If allow_unknown_notations is false, then any notation that does not match a NOTATION_SPECS
    # entry will raise a KeyError.
    allow_unknown_notations = False

    def __init__(self, name, properties, lineno):
        super(Notation, self).__init__()
        self.name = name
        self.raw_properties = properties
        self.lineno = lineno

        self.spec = None
        try:
            self.spec = self.NOTATION_SPECS[name]
        except KeyError:
            if self.allow_unknown_notations:
                print("Warning: Unknown notation: {}".format(name))
            else:
                raise KeyError("Unknown notation: {}".format(name))

        # Only check properties if this is a known notation.
        if self.spec is not None:
            provided_props = set(self.raw_properties.keys())
            allowed_props = {prop.name for prop in self.spec.properties}

            unknown_props = provided_props - allowed_props
            if unknown_props:
                raise KeyError(
                    "Unknown properties for notation {}: {}".format(name, sorted(unknown_props))
                )

            # Parse the property values
            self.properties = {
                prop: self.parse_property(prop, value)
                for prop, value in self.raw_properties.items()
            }

    def parse_property(self, prop_name, raw_value):
        if self.spec is None:
            return
        [prop_spec] = [spec for spec in self.spec.properties if spec.name == prop_name]
        if prop_spec.type == "string":
            if '"' not in raw_value:
                raise KeyError(
                    "Expected a string for notation {} {}: {}".format(
                        self.name,
                        prop_name,
                        raw_value,
                    )
                )
            return raw_value[1:-1]
        elif prop_spec.type == "bool":
            if raw_value not in ("true", "false"):
                raise KeyError(
                    "Expected true or false for notation {} {}: {}".format(
                        self.name,
                        prop_name,
                        raw_value,
                    )
                )
            return dict(true=True, false=False)[raw_value]
        else:
            raise AssertionError("unhandled prop_spec.type: {}".format(prop_spec.type))

    def allowed_on_enum(self):
        return self.spec is None or "enum" in self.spec.allowed

    def allowed_on_struct(self):
        return self.spec is None or "struct" in self.spec.allowed

    def __repr__(self):
        properties = ", ".join(
            "{} = {}".format(key, value) for key, value in self.raw_properties.items()
        )
        return "{} {{ {} }}\n".format(self.name, properties)


class Enum(AstNode):
    """A description of an lcm enum type"""

    @classmethod
    def from_name_and_cases(cls, name, case_names_and_values, type_name="int32_t"):
        value_to_case = {}  # type: T.Dict[int, str]
        for case_name, case_value in case_names_and_values:
            if case_name in value_to_case.values():
                raise KeyError("Case name {} is not unique".format(case_name))
            if case_value in value_to_case:
                existing_name = value_to_case[case_value]
                raise KeyError(
                    'Value {} is not unique: it is used by "{}" and "{}"'.format(
                        case_value,
                        existing_name,
                        case_name,
                    )
                )
            value_to_case[case_value] = case_name

        cases = [
            EnumCase(name=case_name, value_str=str(case_value))
            for case_value, case_name in sorted(iteritems(value_to_case))
        ]
        return cls(
            name=name, type_ref=TypeRef(type_name), cases=cases, notations=[], reserved_ids=[]
        )

    def __init__(self, name, type_ref, cases, notations, reserved_ids):
        super(Enum, self).__init__()
        equivalent_members = [Member(type_ref, "value")]
        equivalent_members += [ConstMember(type_ref, case.name, case.value_str) for case in cases]
        self.equivalent_struct = Struct(name, equivalent_members, [])

        self.option_type_ref = TypeRef("option_t")
        self.value_member = Member(self.option_type_ref, "value")
        self.cases = list(cases)
        self.storage_type_ref = type_ref
        self.type_ref = TypeRef(name)
        self.notations = list(notations)
        self.reserved_ids = set(reserved_ids)

    def __repr__(self):
        notations = "".join(repr(notation) for notation in self.notations)
        children = "\n".join("  " + repr(case) for case in self.cases)
        reserved = ""
        if self.reserved_ids:
            reserved = "  {}\n".format(ReservedFieldGroup(self.reserved_ids))
        return "{}enum {} : {} {{\n{}{}\n}};".format(
            notations,
            self.name,
            self.storage_type_ref,
            reserved,
            children,
        )

    def __eq__(self, other):
        try:
            other_tuple = (other.name, other.storage_type_ref, other.cases)
        except AttributeError:
            return False
        return (self.name, self.storage_type_ref, self.cases) == other_tuple

    def __ne__(self, other):
        return not self == other

    @property
    def name(self):
        # type: () -> str
        return self.type_ref.name

    @property
    def full_name(self):
        # type: () -> str
        return self.type_ref.full_name

    def get_notation(self, name):
        # get the first notation with the given name, else None
        return next((notation for notation in self.notations if notation.name == name), None)

    def get_notation_property(self, name, prop_name):
        notation = self.get_notation(name)
        if notation is None:
            return None
        prop_value = notation.properties.get(prop_name)
        if prop_value is None:
            return None
        return prop_value

    def case_for_int_value(self, int_value):
        for case in self.cases:
            if case.int_value == int_value:
                return case
        raise KeyError("Value {} not found in:\n{}".format(int_value, self))

    def add_package_name(self, package_name):
        self.type_ref.add_package_name(package_name)
        self.equivalent_struct.add_package_name(package_name)
        self.option_type_ref.add_package_name(self.type_ref.full_name)

    def reference_check(self):
        self.equivalent_struct.reference_check()

        disallowed_notations = [x.name for x in self.notations if not x.allowed_on_enum()]
        if disallowed_notations:
            raise ValueError(
                "Invalid notations {} on an enum ({})".format(disallowed_notations, self.name)
            )
        # validate the case values and reservations
        cases_by_id = {}  # type: T.Dict[int, str]
        for case in self.cases:
            if case.int_value in self.reserved_ids:
                raise KeyError(
                    "Enum case {} cannot use reserved id {}".format(case.name, case.int_value)
                )
            if case.int_value in cases_by_id:
                raise KeyError(
                    "Enum case {} reuses id {}. Also used by {}".format(
                        case.name,
                        case.int_value,
                        cases_by_id[case.int_value],
                    )
                )
            cases_by_id[case.int_value] = case.name

    def compute_hash(self):
        return self.equivalent_struct.compute_hash()


class EnumCase(AstNode):
    """A name/value case of an lcm enum type"""

    def __init__(self, name, value_str):
        super(EnumCase, self).__init__()
        self.name = name
        self.value_str = value_str
        self.type_ref = TypeRef(name)

    def __repr__(self):
        return "{} = {},".format(self.name, self.value_str)

    def __eq__(self, other):
        try:
            other_tuple = (other.name, other.value_str)
        except AttributeError:
            return False
        return (self.name, self.value_str) == other_tuple

    def __ne__(self, other):
        return not self == other

    @property
    def int_value(self):
        return int(self.value_str, base=0)


class Struct(AstNode):
    """A description of an lcmtype"""

    def __init__(self, name, members, notations):
        super(Struct, self).__init__()
        self.reserved_ids = [
            field_id
            for group in members
            if isinstance(group, ReservedFieldGroup)
            for field_id in group.field_ids
        ]
        self.members = [member for member in members if not isinstance(member, ReservedFieldGroup)]
        self.member_map = {member.name: member for member in self.members}
        self.type_ref = TypeRef(name)
        self.notations = list(notations)

    def __repr__(self):
        notations = "".join(repr(notation) for notation in self.notations)
        reserved = [ReservedFieldGroup(self.reserved_ids)] if self.reserved_ids else []
        children = "\n".join("  " + repr(member) for member in reserved + self.members)
        return "{}struct {} {{\n{}\n}};".format(notations, self.name, children)

    @property
    def name(self):
        # type: () -> str
        return self.type_ref.name

    @property
    def full_name(self):
        # type: () -> str
        return self.type_ref.full_name

    def get_notation(self, name):
        # get the first notation with the given name, else None
        return next((notation for notation in self.notations if notation.name == name), None)

    def get_notation_property(self, name, prop_name):
        notation = self.get_notation(name)
        if notation is None:
            return None
        prop_value = notation.properties.get(prop_name)
        if prop_value is None:
            return None
        return prop_value

    def add_package_name(self, package_name):
        self.type_ref.add_package_name(package_name)
        for member in self.members:
            member.add_package_name(package_name)

    def reference_check(self):
        for member in self.members:
            member.reference_check(self)

        disallowed_notations = [x.name for x in self.notations if not x.allowed_on_struct()]
        if disallowed_notations:
            raise ValueError(
                "Invalid notations {} on a struct ({})".format(disallowed_notations, self.name)
            )

        has_protobuf_notation = bool(self.get_notation("#protobuf"))

        # NOTE(jeff): Validate field IDs even if the struct isn't protobuf
        # TODO(jeff): add a #no-protobuf notation to disable these checks and activate other checks
        # that disallow field-ids entirely.
        dimension_members = set()
        field_ids = set()
        # check that all non-const members have a unique, positive field_id
        for member in self.members:
            if not isinstance(member, ArrayMember):
                continue
            for dim in member.dims:
                if dim.size_int is not None:
                    continue
                if dim.auto_member:
                    continue
                dim_member = self.member_map[dim.size_str]
                if isinstance(dim_member, ConstMember):
                    continue
                if dim_member.name in dimension_members and has_protobuf_notation:
                    raise KeyError(
                        "multiple members of {} use the same dim member: {}".format(
                            self.name,
                            dim_member.name,
                        )
                    )
                dimension_members.add(dim_member.name)

        for member in self.members:
            if isinstance(member, ConstMember):
                continue
            if member.name in dimension_members:
                if member.field_id is None:
                    continue
                else:
                    # NOTE(jeff): We could probably support this, but it's likely a bug
                    raise KeyError(
                        "member {}.{} cannot be used as a dimension and"
                        "have an id for protobuf".format(
                            self.name,
                            member.name,
                        )
                    )
            # make sure there is an id
            if member.field_id is None:
                if has_protobuf_notation:
                    raise KeyError(
                        "member {}.{} is missing an id required for protobuf".format(
                            self.name,
                            member.name,
                        )
                    )
            elif member.field_id <= 0:
                raise KeyError(
                    "member {}.{} has non-positive field id {}".format(
                        self.name,
                        member.name,
                        member.field_id,
                    )
                )
            elif member.field_id in self.reserved_ids:
                raise KeyError(
                    "member {}.{} has reserved field id {}".format(
                        self.name,
                        member.name,
                        member.field_id,
                    )
                )
            elif member.field_id in field_ids:
                raise KeyError(
                    "member {}.{} has non-unique field id {}".format(
                        self.name,
                        member.name,
                        member.field_id,
                    )
                )
            if member.field_id is not None:
                field_ids.add(member.field_id)

    def compute_hash(self):
        type_hash = Hash()
        for member in self.members:
            if not isinstance(member, ConstMember):
                member.compute_hash(type_hash)
        return type_hash


class ReservedFieldGroup(AstNode):
    """A group of reserved field ids"""

    def __init__(self, field_ids):
        super(ReservedFieldGroup, self).__init__()
        self.field_ids = set(field_ids)

    def __repr__(self):
        return "reserved {};".format(", ".join(str(fid) for fid in sorted(self.field_ids)))


class Member(AstNode):
    """A field of an lcmtype"""

    def __init__(self, type_ref, name, field_id=None, comments=None):
        super(Member, self).__init__()
        self.name = name
        self.type_ref = type_ref
        self.field_id = field_id
        self.comments = comments or []

    @property
    def ndim(self):
        return 0

    def __repr__(self):
        if self.field_id is None:
            return "{} {};".format(self.type_ref, self.name)
        return "{} {} = {};".format(self.type_ref, self.name, self.field_id)

    def compute_hash(self, type_hash):
        self.compute_hash_prefix_for_auto_members(type_hash)

        # Hash the member name
        type_hash.update_string(self.name)

        # If the member is a primitive type, include the type
        # signature in the hash. Do not include them for compound
        # members, because their contents will be included, and we
        # don't want a struct's name change to break the hash.
        if self.type_ref.is_primitive_type():
            type_hash.update_string(self.type_ref.name)

        self.compute_hash_for_dimensions(type_hash)

    def compute_hash_prefix_for_auto_members(self, type_hash):
        # Normal members don't have dimensions, thus don't have virtual members to add to the hash
        pass

    def compute_hash_for_dimensions(self, type_hash):
        # Normal members don't have dimensions
        type_hash.update(0)

    def add_package_name(self, package_name):
        self.type_ref.add_package_name(package_name)

    def reference_check(self, _):
        pass


class TypeRef(AstNode):
    """A named reference to an existing type"""

    def __init__(self, ref_str):
        super(TypeRef, self).__init__()
        if ref_str in PRIMITIVE_TYPES:
            # This is a primitive.
            self.package_name = None
            self.name = ref_str
        elif "." in ref_str:
            # This is already a fully-qualified type.
            self.package_name, self.name = ref_str.split(".")
        else:
            # The package name is implied and will be filled in later with add_package_name()
            self.package_name = "<PACKAGE-NOT-SET>"
            self.name = ref_str

    def add_package_name(self, package_name):
        if self.package_name == "<PACKAGE-NOT-SET>":
            # We only set the package name if the given one was implied.
            self.package_name = package_name

    @property
    def full_name(self):
        if self.package_name:
            return "{}.{}".format(self.package_name, self.name)
        else:
            return self.name

    def __repr__(self):
        return self.full_name

    def __eq__(self, other):
        try:
            other_tuple = (other.name, other.package_name)
        except AttributeError:
            return False
        return (self.name, self.package_name) == other_tuple

    def __ne__(self, other):
        return not self == other

    def is_non_string_primitive_type(self):
        # Many languages treat strings differently than other primitives, so this is a common check.
        return self.name != "string" and self.is_primitive_type()

    def is_primitive_type(self):
        return self.name in PRIMITIVE_TYPES

    def is_numeric_type(self):
        return self.name in NUMERIC_TYPES

    def is_const_type(self):
        return self.name in CONST_TYPES


class ArrayDim(AstNode):
    """A static or dynamic size of an array in a single axis"""

    def __init__(self, sizes):
        super(ArrayDim, self).__init__()
        self.sizes_as_declared = sizes  # This is a tuple of the 0-2 strings in the .lcm definition.
        # These will be set after the reference check.
        self.size_str = None
        self.size_int = None
        self._dynamic = None
        self._auto_member = None

    @property
    def dynamic(self):
        """Return True if the dimension is dynamic based on struct contents.
        This value is not known until after struct.reference_check(...) is called."""
        if self._dynamic is None:
            raise RuntimeError("Cannot determine until reference_check() is called")
        return self._dynamic

    @property
    def auto_member(self):
        """Return the virtual Member of the dimension if using automatic length encoding.
        This value is not known until after struct.reference_check(...) is called."""
        if self._dynamic is None:
            raise RuntimeError("Cannot determine until reference_check() is called")
        return self._auto_member

    def compute_hash(self, type_hash):
        type_hash.update(1 if self.dynamic else 0)
        type_hash.update_string(self.size_str)

    def reference_check(self, struct, member):
        """
        Use the references to determine if this arraydim is static or dynamic, and to build the
        rest of the info. There are 6 valid ways of specifying an array dimension:

        1) Literal Integer (e.g. [3]):
                Specify a fixed array with a literal integer value.

        2) Const Integer Member (e.g. [ITEM_SIZE]):
                Specify a fixed array with the name of a const integer member of this struct.

        3) Integer Member (e.g. [item_count]):
                Specify a dynamic array with the name of a integer member defined in this struct
                before this dimension's member.

        4) Integer Type and Name (e.g. [uint16_t item_count])
                Specify a auto dynamic array with the given integer type and name.

        5) Integer Type (e.g. [uint16_t]):
                Specify a auto dynamic array with the given integer type and the default name.
                i.e. [uint16_t num_<name>]

        6) Empty Brackets (e.g. []):
                Specify a auto dynamic array with default options.
                i.e. [int32_t num_<name>]
        """
        size_declaration = "[{}]".format(" ".join(self.sizes_as_declared))
        if len(self.sizes_as_declared) == 1:
            # Get case 1 out of the way ASAP
            try:
                self.size_int = int(self.sizes_as_declared[0], base=0)
                self.size_str = self.sizes_as_declared[0]
                self._dynamic = False
                return  # case 1
            except ValueError:
                # The size is not a simple integer
                pass

        # collapse case 6 into case 5
        decl_tuple = self.sizes_as_declared or ("int32_t",)  # type: T.Tuple[str, ...]
        # collapse case 5 into case 4
        if len(decl_tuple) == 1 and decl_tuple[0] in INTEGER_TYPES:
            auto_name = "num_{}".format(member.name)
            decl_tuple = (decl_tuple[0], auto_name)
        # Now decl_tuple is len=1 for cases 2 and 3, and len=2 for cases 4, 5, and 6

        if len(decl_tuple) == 2:
            # auto-member version
            type_name, member_name = decl_tuple

            if member.ndim > 1:
                raise TypeError(
                    "ArrayDim {} with auto-size is not allowed on "
                    "multi-dimensional arrays".format(size_declaration)
                )

            if type_name not in INTEGER_TYPES:
                raise TypeError(
                    "ArrayDim {} does not specify a valid integer type: {}".format(
                        size_declaration,
                        type_name,
                    )
                )

            # Confirm that the virtual member name is NOT a member of this struct.
            if member_name in struct.member_map:
                raise NameError(
                    "ArrayDim {} with auto-size conflicts with existing member {}".format(
                        size_declaration,
                        member_name,
                    )
                )

            self._dynamic = True
            self.size_int = None
            self.size_str = member_name
            self._auto_member = Member(TypeRef(type_name), member_name)
            return  # cases 4, 5, 6

        assert len(decl_tuple) == 1
        member_name = decl_tuple[0]
        # Confirm that the size is a reference to a member of this struct.
        if member_name not in struct.member_map:
            raise NameError("ArrayDim {} is not a defined member".format(size_declaration))

        size_member = struct.member_map[member_name]
        if size_member.type_ref.name not in INTEGER_TYPES:
            raise TypeError("ArrayDim {} is not a valid array type".format(size_declaration))

        # Determine if this member is a const, or a regular member
        if isinstance(size_member, ConstMember):
            # This references a ConstMember, so we can use its value directly in templates.
            self._dynamic = False
            self.size_int = size_member.value
            # Convert the size back to string for use in templates.
            # From a template's perspective, this dim is identical to a simple integer size.
            self.size_str = str(self.size_int)
            return  # case 2

        # Check that the non-static member appears before the array
        if struct.members.index(size_member) >= struct.members.index(member):
            raise ValueError(
                "ArrayDim {} must appear before the array {}".format(member_name, member.name)
            )

        # This references a variable in the message whose value is not known statically.
        self._dynamic = True
        self.size_int = None
        self.size_str = member_name
        return  # case 3

    def __repr__(self):
        return "[{}]".format(" ".join(self.sizes_as_declared))


class ArrayMember(Member):
    """A single-type container with one or more dimensions of fixed and/or dynamic size."""

    # TODO(matt): might be better to just combine with Member, as lcmgen original does.

    def __init__(self, type_ref, name, dims, field_id=None):
        super(ArrayMember, self).__init__(type_ref, name, field_id=field_id)
        self.dims = dims

    def compute_hash_prefix_for_auto_members(self, type_hash):
        # this adds the hash of virtual fields would have been used by a manual dynamic array
        for dim in self.dims:
            if dim.auto_member:
                dim.auto_member.compute_hash(type_hash)

    def compute_hash_for_dimensions(self, type_hash):
        # hash the dimensionality information
        type_hash.update(len(self.dims))
        for dim in self.dims:
            dim.compute_hash(type_hash)

    def __repr__(self):
        dims_str = "".join(repr(dim) for dim in self.dims)
        if self.field_id is None:
            return "{} {}{};".format(self.type_ref, self.name, dims_str)
        return "{} {}{} = {};".format(self.type_ref, self.name, dims_str, self.field_id)

    def is_constant_size(self):
        return not any(dim.dynamic for dim in self.dims)

    def reference_check(self, struct):
        for dim in self.dims:
            dim.reference_check(struct, self)

    @property
    def ndim(self):
        return len(self.dims)


class ConstMember(Member):
    """An attribute whose value is bound to the type itself, not encoded in a message"""

    def __init__(self, type_ref, name, value_str):
        super(ConstMember, self).__init__(type_ref, name)
        if not type_ref.is_const_type():
            raise TypeError(
                "Constant '{}' from line {} must be one of {}. '{}' found.".format(
                    name,
                    type_ref.lineno,
                    CONST_TYPES,
                    type_ref.name,
                )
            )
        try:
            self.value = CONST_TYPE_MAP[type_ref.name](value_str)
        except ValueError:
            print("Error parsing const type {}.".format(self.name))
            raise
        self.value_str = value_str

    def __repr__(self):
        return "const {} {} = {};".format(self.type_ref, self.name, self.value_str)
