# lextab.py. This file automatically created by PLY (version 3.11). Don't edit!
# type: ignore

_tabversion = "3.10"
_lextokens = set(
    (
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
)
_lexreflags = 64
_lexliterals = ""
_lexstateinfo = {"INITIAL": "inclusive"}
_lexstatere = {
    "INITIAL": [
        (
            '(?P<t_CLOSE_BRACE>})|(?P<t_COMMENT>\\/\\/(.*?)[\\n])|(?P<t_CONST>const\\W)|(?P<t_ENUM>enum\\W)|(?P<t_EXTENDED_COMMENT>/\\*((.*?[\\n]*)*?)\\*/)|(?P<t_PACKAGE>package\\W)|(?P<t_STRUCT>struct\\W)|(?P<t_WHITESPACE>[ \n\t]+)|(?P<t_RESERVED>reserved\\W)|(?P<t_SEMICOLON>;[ \\t]*(\\/\\/[^\\n]*)?)|(?P<t_MACRO>[#][a-zA-Z._0-9-]+)|(?P<t_WORD>[a-zA-Z._0-9-]+)|(?P<t_STRING>"[^"]+")|(?P<t_CLOSE_BRACKET>\\])|(?P<t_OPEN_BRACKET>\\[)|(?P<t_COLON>:)|(?P<t_COMMA>,)|(?P<t_EQUALS>=)|(?P<t_OPEN_BRACE>{)',
            [
                None,
                ("t_CLOSE_BRACE", "CLOSE_BRACE"),
                ("t_COMMENT", "COMMENT"),
                None,
                ("t_CONST", "CONST"),
                ("t_ENUM", "ENUM"),
                ("t_EXTENDED_COMMENT", "EXTENDED_COMMENT"),
                None,
                None,
                ("t_PACKAGE", "PACKAGE"),
                ("t_STRUCT", "STRUCT"),
                ("t_WHITESPACE", "WHITESPACE"),
                ("t_RESERVED", "RESERVED"),
                ("t_SEMICOLON", "SEMICOLON"),
                None,
                ("t_MACRO", "MACRO"),
                ("t_WORD", "WORD"),
                ("t_STRING", "STRING"),
                (None, "CLOSE_BRACKET"),
                (None, "OPEN_BRACKET"),
                (None, "COLON"),
                (None, "COMMA"),
                (None, "EQUALS"),
                (None, "OPEN_BRACE"),
            ],
        )
    ]
}
_lexstateignore = {"INITIAL": ""}
_lexstateerrorf = {"INITIAL": "t_error"}
_lexstateeoff = {}
