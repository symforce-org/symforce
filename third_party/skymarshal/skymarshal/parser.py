# aclint: py2 py3
# mypy: allow-untyped-defs
# LCM Defintion File Parsing
# ==========================
# Here we rely on PLY (python-lex-yacc) to parse a stream of tokens into an abstract syntax tree.
# PLY supports yacc-like syntax and uses fancy python introspection to build its parse table.
# Functions are required to have p_ prefix and docstrings that explain the syntax rules.
# During parsing, the parse stack matches a rule, the corresponding function is executed.
# A function's p argument is the list of productions in the rule, with p[0] being the output.
from __future__ import absolute_import, print_function

from ply import yacc

from . import syntax_tree

# PLY relies on the existence of these names for parsing
from .tokenizer import tokens  # pylint: disable=unused-import
from .tokenizer import lexer


def recurse_left(p, spaces=0, extend=False):
    """items :
    | item
    | items <spaces> item"""
    # Helper function for recursively defined productions.
    if len(p) == 1:
        # This is an empty sequence. Create the empty list.
        p[0] = []
    elif len(p) == 2:
        # This is the first element in a sequence. Create a 1-element list.
        p[0] = [p[1]]
    else:
        # This is a an element after an existing list. Append.
        item = p[2 + spaces]
        if extend:
            p[1].extend(item)
        else:
            p[1].append(item)
        p[0] = p[1]


# This is the top-level rule: return a list of packages
def p_packages(p):
    """packages : packages package
    |"""
    recurse_left(p)


# Each package has an identifer and a list of type_definitions
def p_package(p):
    """package : PACKAGE identifier SEMICOLON type_definitions"""
    p[0] = syntax_tree.Package(name=p[2], type_definitions=p[4])


# An identifer is just a word token (see lexer.py)
def p_identifier(p):
    "identifier : WORD"
    p[0] = p[1][1]


# A close_brace_semi is a CLOSE_BRACE optionally followed by a SEMICOLON
def p_close_brace_semi(p):
    """close_brace_semi : CLOSE_BRACE
    | CLOSE_BRACE SEMICOLON"""
    p[0] = p[1]


# Each enum has an identifier, optional type, and enum_contents
def p_enum(p):
    """enum : notations ENUM identifier COLON type OPEN_BRACE enum_contents close_brace_semi
    | notations ENUM identifier OPEN_BRACE enum_contents close_brace_semi"""
    if len(p) == 9:
        type_ref = p[5]
        reserved_ids, cases = p[7]
    elif len(p) == 7:
        type_ref = syntax_tree.TypeRef("int32_t")
        reserved_ids, cases = p[5]
    else:
        raise AssertionError("parser bug: len(p)={}".format(len(p)))
    p[0] = syntax_tree.Enum(
        name=p[3], type_ref=type_ref, cases=cases, notations=p[1], reserved_ids=reserved_ids
    )
    p[0].comments = p[2]
    p[0].reference_check()


def p_notations(p):
    """notations : notations notation
    |"""
    recurse_left(p)


def p_notation(p):
    """notation : MACRO OPEN_BRACE notation_properties CLOSE_BRACE
    | MACRO OPEN_BRACE CLOSE_BRACE
    | MACRO"""
    if len(p) == 5:
        p[0] = syntax_tree.Notation(name=p[1][1], properties=dict(p[3]), lineno=p[1][0])
    else:
        p[0] = syntax_tree.Notation(name=p[1][1], properties=dict(), lineno=p[1][0])


# Assignments are comma-delimited and cannot be empty
def p_notation_properties(p):
    """notation_properties : notation_property
    | notation_properties COMMA notation_property"""
    recurse_left(p, 1)


# A single assignment has a equals sign
def p_notation_property(p):
    """notation_property : identifier EQUALS const_value
    | identifier EQUALS const_string"""
    p[0] = (p[1], p[3])


# enum_contents are any number of reserved statements, followed by enum values
def p_enum_contents(p):
    """enum_contents : reserved_groups enum_values
    | reserved_groups enum_values COMMA"""
    p[0] = p[1], p[2]


# The enum_values is a list of 1 or more comma separated 'enum_value's
def p_enum_values(p):
    """enum_values : enum_values COMMA enum_value
    | enum_value"""
    recurse_left(p, spaces=1)


# enum_value is an identifier and an integer.
def p_enum_value(p):
    """enum_value : identifier EQUALS WORD"""
    p[0] = syntax_tree.EnumCase(name=p[1], value_str=p[3][1])


# A struct list contains zero or more structs
def p_type_definitions(p):
    """type_definitions : type_definitions struct
    | type_definitions enum
    |"""
    recurse_left(p)


# Each struct has an identifier and members
def p_struct(p):
    """struct : notations STRUCT identifier OPEN_BRACE members close_brace_semi"""
    p[0] = syntax_tree.Struct(name=p[3], members=p[5], notations=p[1])
    p[0].comments = p[2]
    p[0].reference_check()


# The members list is created by pulling individual members out of "type groups"
def p_members(p):
    """members : members type_group
    |"""
    recurse_left(p, extend=True)


# A type group is a list of one or more members that share the same type.
# For example: int32_t foo, bar;
def p_member_type_group(p):
    """type_group : type member_names SEMICOLON"""
    # Handle the definition of multiple members on a single line.
    type_ref = p[1]
    names = p[2]
    comments = p[3]
    p[0] = [syntax_tree.Member(type_ref=type_ref, name=name, comments=comments) for name in names]


# A type group with id is one member.
# For example: int32_t foo = 1;
def p_member_with_id_type_group(p):
    """type_group : type identifier EQUALS field_id SEMICOLON"""
    # Handle the definition of multiple members on a single line.
    type_ref = p[1]
    name = p[2]
    field_id = p[4]
    comments = p[5]
    p[0] = [syntax_tree.Member(type_ref=type_ref, name=name, field_id=field_id, comments=comments)]


def p_member_reserved_group(p):
    """type_group : RESERVED field_ids SEMICOLON"""
    p[0] = [syntax_tree.ReservedFieldGroup(p[2])]


def p_reserved_groups(p):
    """reserved_groups : reserved_groups reserved_group
    |"""
    recurse_left(p, extend=True)


def p_reserved_group(p):
    """reserved_group : RESERVED field_ids SEMICOLON"""
    p[0] = p[2]


def p_field_ids(p):
    """field_ids : field_id
    | field_ids COMMA field_id"""
    recurse_left(p, spaces=1)


def p_field_id(p):
    """field_id : WORD"""
    p[0] = int(p[1][1])  # Drop the line number and convert to int
    assert p[0] > 0, "explicit field_id must be a postitive integer"


# types use the same WORD token as identifiers
def p_type(p):
    """type : WORD"""
    type_ref = syntax_tree.TypeRef(p[1][1])
    type_ref.lineno = p[1][0]
    p[0] = type_ref


# Member names are comma-delimited and cannot be empty
def p_member_names(p):
    """member_names : identifier
    | member_names COMMA identifier"""
    recurse_left(p, 1)


# Const member declarations are similar, but have assignments instead of just names
def p_const_member_type_group(p):
    """type_group : CONST type const_member_assignments SEMICOLON"""
    # Handle the definition of multiple const members on a single line.
    type_ref = p[2]
    assignments = p[3]
    p[0] = [syntax_tree.ConstMember(type_ref, name, value) for name, value in assignments]


# Assignments are comma-delimited and cannot be empty
def p_const_member_assignments(p):
    """const_member_assignments : assignment
    | const_member_assignments COMMA assignment"""
    recurse_left(p, 1)


# A single assignment has a equals sign
def p_assignment(p):
    """assignment : identifier EQUALS const_value"""
    p[0] = (p[1], p[3])


# Treat values like words
# TODO(matt): might be better to enforce a number during lexing
def p_const_value(p):
    """const_value : WORD"""
    p[0] = p[1][1]  # ignores the line number at p[1][0]
    # TODO(matt): check the type here?


def p_const_string(p):
    """const_string : STRING"""
    p[0] = p[1]


# A single member can be an array with one or more dimensions.
def p_array_member_type_group(p):
    """type_group : type identifier array_dims SEMICOLON"""
    p[0] = [syntax_tree.ArrayMember(type_ref=p[1], name=p[2], dims=p[3])]


# allow members with ids (for protobuf) to have a single dimension (e.g. repeated)
def p_array_member_with_id_type_group(p):
    """type_group : type identifier array_dim EQUALS field_id SEMICOLON"""
    p[0] = [syntax_tree.ArrayMember(type_ref=p[1], name=p[2], dims=[p[3]], field_id=p[5])]


# A non-empty list of dimensions
def p_array_dims(p):
    """array_dims : array_dim
    | array_dims array_dim"""
    recurse_left(p)


# A single dimension
def p_array_dim(p):
    """array_dim : OPEN_BRACKET CLOSE_BRACKET
    | OPEN_BRACKET WORD CLOSE_BRACKET
    | OPEN_BRACKET WORD WORD CLOSE_BRACKET"""
    if len(p) == 3:
        # Empty brackets
        p[0] = syntax_tree.ArrayDim(tuple())
    else:
        # Either 1 or 2 words in the brackets, passed through as a tuple.
        # May be a literal, a field name, an integer type, or a type and a virtual field name
        size_declaration = tuple(word for _, word in p[2:-1])
        p[0] = syntax_tree.ArrayDim(size_declaration)
        p[0].lineno = p[2][0]


# This function gets called by the parser on error
def p_error(token):
    # NOTE: Call yacc.errok() to proceed with parsing.
    # TODO(matt): add more error handling.
    if token:
        raise LcmParseError(
            'Unable to parse starting from "{}" on line {}'.format(token.value, token.lineno)
        )
    else:
        raise LcmParseError("Unexpected end of input")


PARSER = None


def lcmparse(src, verbose=True, cache=False, debug_src_path=None, allow_unknown_notations=False):
    """Parse an LCM definition source into a list of packages"""
    global PARSER  # pylint: disable=global-statement
    lexer.lineno = 1  # reset the line number on repeat calls to lcmgen

    kwargs = dict(debug=False, write_tables=False)
    if not verbose:
        # will stop any logging of warnings
        kwargs["errorlog"] = yacc.NullLogger()

    if cache and PARSER:
        parser = PARSER
    else:
        # Introspect this file so-far and create the parser
        parser = yacc.yacc(**kwargs)

    if cache:
        PARSER = parser

    syntax_tree.Notation.allow_unknown_notations = allow_unknown_notations

    try:
        packages = parser.parse(src)
    except LcmParseError as ex:
        if debug_src_path:
            raise LcmParseError("{} of {}".format(str(ex), debug_src_path))
        else:
            raise LcmParseError("{}\n==========\n{}\n==========".format(str(ex), src))

    # NOTE(matt): this does not perform deduplication on package or struct names.
    return packages


class LcmParseError(ValueError):
    pass
