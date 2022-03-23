# lextab.py. This file automatically created by PLY (version 3.9). Don't edit!
# type: ignore

_tabversion = "3.8"
_lextokens = set(
    (
        "CLOSE_BRACKET",
        "WORD",
        "CONST",
        "STRING",
        "SEMICOLON",
        "PACKAGE",
        "MACRO",
        "ENUM",
        "RESERVED",
        "EQUALS",
        "OPEN_BRACKET",
        "OPEN_BRACE",
        "COMMA",
        "CLOSE_BRACE",
        "COLON",
        "STRUCT",
    )
)
_lexreflags = 0
_lexliterals = ""
_lexstateinfo = {"INITIAL": "inclusive"}
_lexstatere = {
    "INITIAL": [
        (
            '(?P<t_CLOSE_BRACE>})|(?P<t_WHITESPACE>[ \n\t]+)|(?P<t_ENUM>enum\\W)|(?P<t_PACKAGE>package\\W)|(?P<t_COMMENT>\\/\\/(.*?)[\\n])|(?P<t_STRUCT>struct\\W)|(?P<t_EXTENDED_COMMENT>/\\*((.*?[\\n]*)*?)\\*/)|(?P<t_CONST>const\\W)|(?P<t_RESERVED>reserved\\W)|(?P<t_SEMICOLON>;[ \\t]*(\\/\\/[^\\n]*)?)|(?P<t_MACRO>[#][a-zA-Z._0-9-]+)|(?P<t_WORD>[a-zA-Z._0-9-]+)|(?P<t_STRING>"[^"]+")|(?P<t_OPEN_BRACKET>\\[)|(?P<t_CLOSE_BRACKET>\\])|(?P<t_COLON>:)|(?P<t_COMMA>,)|(?P<t_OPEN_BRACE>{)|(?P<t_EQUALS>=)',
            [
                None,
                ("t_CLOSE_BRACE", "CLOSE_BRACE"),
                ("t_WHITESPACE", "WHITESPACE"),
                ("t_ENUM", "ENUM"),
                ("t_PACKAGE", "PACKAGE"),
                ("t_COMMENT", "COMMENT"),
                None,
                ("t_STRUCT", "STRUCT"),
                ("t_EXTENDED_COMMENT", "EXTENDED_COMMENT"),
                None,
                None,
                ("t_CONST", "CONST"),
                ("t_RESERVED", "RESERVED"),
                ("t_SEMICOLON", "SEMICOLON"),
                None,
                ("t_MACRO", "MACRO"),
                ("t_WORD", "WORD"),
                ("t_STRING", "STRING"),
                (None, "OPEN_BRACKET"),
                (None, "CLOSE_BRACKET"),
                (None, "COLON"),
                (None, "COMMA"),
                (None, "OPEN_BRACE"),
                (None, "EQUALS"),
            ],
        )
    ]
}
_lexstateignore = {"INITIAL": ""}
_lexstateerrorf = {"INITIAL": "t_error"}
_lexstateeoff = {}
