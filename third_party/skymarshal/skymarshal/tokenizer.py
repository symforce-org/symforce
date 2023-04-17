# aclint: py2 py3
# mypy: allow-untyped-defs
# LCM Definition File tokenizer
# We use PLY and regexes to convert a byte stream into a sequence of tokens.
from __future__ import absolute_import, print_function

import os
import shutil
import sys
import tempfile
import typing as T
from functools import wraps

from ply import lex

# Sorry pylint, PLY expects certain names in the module.
tokens = (
    "CLOSE_BRACE",
    "CLOSE_BRACKET",
    "COLON",
    "COMMA",
    "CONST",
    "ENUM",
    "EQUALS",
    "MACRO",
    "OPEN_BRACE",
    "OPEN_BRACKET",
    "PACKAGE",
    "RESERVED",
    "SEMICOLON",
    "STRING",
    "STRUCT",
    "WORD",
)


# Simple tokens without extra actions
t_CLOSE_BRACKET = r"\]"
t_COLON = ":"
t_COMMA = ","
t_EQUALS = "="
t_OPEN_BRACE = "{"
t_OPEN_BRACKET = r"\["

# The remaining tokens are definited using the special functions below.
# PLY looks for functions prefixed with t_ and uses their docstrings as regex patterns.
# When a pattern matches, the function is called with a token object
# The function can modify or remove the token.

# A note on comments in lcm defintion files:
# lcm-gen tracks comments and associates them with structs
# try_parse_and_consume_comment calls are littered throughout that code.
# I haven't figured out an elegant way to do this in PLY's LR parser yet.
# So for now I create a COMMENT_QUEUE which holds unprocessed comments
# When a STRUCT, PACKAGE, CLOSE_BRACE, or SEMICOLOR token is found, the queue gets processed.
# This enables the parser to associate comments without complicating the syntax rules.
COMMENT_QUEUE = []  # type: T.List[str]


def consume_comment_queue(f):
    "Decorator to attach the comments in the queue to the current token"
    # PLY depends on the exact function name and doctring, so we need to copy them with "wraps"
    @wraps(f)
    def f_with_line_counting(t):
        # Put the existing comments on this token for later access
        t.value = list(COMMENT_QUEUE)
        # Clear the comment queue
        COMMENT_QUEUE[:] = []
        return f(t)

    return f_with_line_counting


def count_lines(f):
    "Decorator to count newlines in a token and increment the line number"
    # PLY depends on the exact function name and doctring, so we need to copy them with "wraps"
    @wraps(f)
    def f_with_line_counting(t):
        t.lexer.lineno += sum(1 for c in t.value if c == "\n")
        return f(t)

    return f_with_line_counting


@consume_comment_queue
def t_CLOSE_BRACE(t):
    "}"
    return t


@count_lines
def t_COMMENT(t):
    r"\/\/(.*?)[\n]"
    # Assumes 2 or more leading slashes, strips whitespace.
    text = t.value.lstrip("/").strip()
    # Saves the comment (if non-empty) and makes it accessible to a future token.
    if text:
        COMMENT_QUEUE.append(text)
    # Prevent the comment from entering the token sequence by returning None
    return None


@count_lines
def t_CONST(t):
    r"const\W"
    return t


@count_lines
@consume_comment_queue
def t_ENUM(t):
    r"enum\W"
    return t


@count_lines
def t_EXTENDED_COMMENT(t):
    r"/\*((.*?[\n]*)*?)\*/"
    # Assumes /* text */ and one or more lines.
    comment_body = t.value[2:-2]
    lines = comment_body.splitlines()
    for line in lines:
        # Remove leading stars, and trim whitespace
        text = line.strip().lstrip("*").strip()
        if text:
            # Saves comment lines and makes them accessible to a future token
            COMMENT_QUEUE.append(text)
    # Prevent the comment from entering the token sequence by returning None
    return None


@count_lines
@consume_comment_queue
def t_PACKAGE(t):
    r"package\W"
    return t


def t_RESERVED(t):
    r"reserved\W"
    return t


@count_lines
@consume_comment_queue
def t_STRUCT(t):
    r"struct\W"
    return t


def t_SEMICOLON(t):
    r";[ \t]*(\/\/[^\n]*)?"
    # Copy the queue.
    comments = list(COMMENT_QUEUE)
    # Clear the queue.
    COMMENT_QUEUE[:] = []
    # Get the trailing comment, if any, without the leading semicolon or slashes.
    trailing = t.value[1:].strip().lstrip("/").lstrip()
    if trailing:
        comments.append(trailing)
    # Assign to the value for later retrieval during parsing.
    t.value = comments
    return t


def t_MACRO(t):
    r"[#][a-zA-Z._0-9-]+"
    # Attach the current line number to the macro so we can access it later for debugging
    t.value = (t.lexer.lineno, t.value)
    return t


def t_WORD(t):
    "[a-zA-Z._0-9-]+"
    # Attach the current line number to the word so we can access it later for debugging
    t.value = (t.lexer.lineno, t.value)
    return t


def t_STRING(t):
    r'"[^"]+"'
    return t


@count_lines
def t_WHITESPACE(_):
    "[ \n\t]+"
    # Drop all whitespace tokens from the sequence
    return None


def t_error(arg):
    print("error", arg)


if sys.version_info.major < 3:
    lextab_opt = "lextab_py2"
else:
    lextab_opt = "lextab_py3"


# Create the lexer (this lowercase name is required by PLY)
if os.environ.get("SKYMARSHAL_REGENERATE_LEXER"):
    lexer = lex.lex(optimize=1)  # this will create lextab.py
else:
    tempdir = tempfile.mkdtemp()
    try:
        lexer = lex.lex(optimize=1, lextab=lextab_opt, outputdir=tempdir)
    finally:
        shutil.rmtree(tempdir)


def generate_tokens(src):
    "Yield a sequence of tokens"
    lexer.input(src)
    for t in lexer:
        yield t.type
    # Should we reset?
    lexer.lineno = 1


def debug_tokens(src):
    "Print the tokens to stdout"
    print("\n".join(generate_tokens(src)))
