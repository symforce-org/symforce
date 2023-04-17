# aclint: py2 py3
from __future__ import absolute_import, print_function

import sys
import typing as T

# Support both python 2 and 3 versions of the string module
if sys.version_info[0] == 2:
    # In python2.7, it's called uppercase
    from string import uppercase
else:
    # ...but in python3, it's called ascii_uppercase. So rename it.
    from string import ascii_uppercase as uppercase

StrType = T.TypeVar("StrType", str, T.Text)


def snakecase_to_camelcase(snake_string):
    # type: (StrType) -> StrType
    return "".join(word.capitalize() for word in snake_string.split("_"))


def snakecase_to_lower_camelcase(snake_string):
    # type: (StrType) -> StrType
    words = snake_string.split("_")
    return words[0] + "".join(word.capitalize() for word in words[1:])


def camelcase_to_snakecase(camelcase, to_upper=False):
    # type: (StrType, bool) -> StrType
    if "_" in camelcase:
        # This string is already using underscores.
        if to_upper:
            return camelcase.upper()
        else:
            return camelcase.lower()
    out = []  # type: T.List[StrType]
    for char in camelcase:
        if out and char in uppercase:
            out.append("_")
        if to_upper:
            out_char = char.upper()
        else:
            out_char = char.lower()
        out.append(out_char)
    return "".join(out)


def is_camelcase(string):
    # type: (StrType) -> bool
    if not string:
        return False
    if "_" in string or " " in string:
        return False
    return string[0] == string[0].upper()
