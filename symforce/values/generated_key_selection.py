# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
The C++ Values object uses fixed-size Key structs that contain a char with integer subscript and superscript. SymPy symbols can have arbitrary string names. This module contains heuristics to map from string symbol names to Key types that are reasonably intuitive and debuggable. Some of these heuristics include:

Works best if the symbol names are snake case, although supports non-snake-case or strings that
aren't variable names.  General strategy is to try the following for each name, until we find a key
that is available to pick:

    1) The first letter of each word (words defined by separating on underscores)
    2) All the other characters in the name

For each character we try, we'll prefer the lowercase version, and then the uppercase version
if lowercase is taken.  If the name has a number in it, we'll first try using that as the
subscript, or won't use a subscript if there's already a key with the same letter and
subscript.  For example:

.. code-block:: python

    pick_generated_keys_for_variable_names(['foo', 'foo2', 'foo_bar', 'foo_bar2', 'foo_baz'])

returns

.. code-block:: python

    {
        'foo': ('f', None),
        'foo2': ('f', 2),
        'foo_bar': ('F', None),
        'foo_bar2': ('F', 2),
        'foo_baz': ('b', None)
    }

For a more thorough example, see symforce_values_generated_key_selection_test.py
"""

import re
import string

from symforce import typing as T


class GeneratedKey(T.NamedTuple):
    """
    A Key to generate, with a single letter and optional subscript
    """

    letter: str
    sub: T.Optional[int] = None


def _choices_for_name(name: str) -> T.Tuple[T.List[str], T.Optional[int]]:
    """
    For a symbol name (similar to a Python variable name), return
    1) A list of letters to consider, in order of preference
    2) A subscript to use (the first integer in name if it exists)
    """
    name = name.lower()

    # Strip everything that's not _ or a lowercase letter, then split
    name_parts = re.sub("[^a-z_]+", "", name).split("_")

    # Find the first integer in the name if it exists
    match = re.search("-?[0-9]+", name)
    if match is None:
        sub = None
    else:
        sub = int(match.group(0))

    letters_to_try = []

    # Try the first letter of each word
    for part in name_parts:
        if not part:
            continue
        letters_to_try.append(part[0])

    # Try all the other letters in the name
    for part in name_parts:
        for letter in part[1:]:
            letters_to_try.append(letter)

    # Remove duplicates.  set is not order-preserving, but dict is
    # https://stackoverflow.com/a/53657523
    letters_to_try = list(dict.fromkeys(letters_to_try).keys())

    letters_to_try += [l for l in string.ascii_lowercase if l not in letters_to_try]

    return letters_to_try, sub


def _pick_key_for_choices(
    letters_to_try: T.List[str],
    sub: T.Optional[int],
    is_unused: T.Callable[[str, T.Optional[int]], bool],
    next_unused_sub: T.Callable[[str], int],
) -> GeneratedKey:
    """
    Given a list of letters to try in order of preference, and an optional digit, pick a
    GeneratedKey
    """
    assert letters_to_try, "letters_to_try should not be empty"

    for letter in letters_to_try:
        if is_unused(letter, sub):
            return GeneratedKey(letter, sub)
        if is_unused(letter.upper(), sub):
            return GeneratedKey(letter.upper(), sub)

        if sub is not None:
            # Also try without the sub
            if is_unused(letter, None):
                return GeneratedKey(letter, None)
            if is_unused(letter.upper(), None):
                return GeneratedKey(letter.upper(), None)

    # Just pick the first preferred letter, with a higher subscript
    letter = letters_to_try[0]
    return GeneratedKey(letter, next_unused_sub(letter))


def pick_generated_keys_for_variable_names(
    names: T.Sequence[str], excluded_keys: T.Set[GeneratedKey] = None
) -> T.Dict[str, GeneratedKey]:
    """
    Pick a character (and possibly a subscript) to represent each string in names

    See module docstring for the heuristics used to pick characters and subscripts

    Args:
        names: List of strings to generate keys for
        excluded_keys: Set of GeneratedKeys to exclude from the allowed keys
    """

    def mark_used(key: GeneratedKey) -> None:
        if key.letter not in used_keys:
            used_keys[key.letter] = set()
        used_keys[key.letter].add(key.sub)

    def is_unused(letter: str, sub: T.Optional[int]) -> bool:
        if letter not in used_keys:
            return True
        if sub not in used_keys[letter]:
            return True
        return False

    def next_unused_sub(letter: str) -> int:
        """
        Pick the subscript either one more than the current max used, or 0 if the current max is
        None or <0
        """
        assert used_keys[letter]
        used_subscripts = T.cast(
            T.List[int], list(filter(lambda s: s is not None, used_keys[letter]))
        )
        if used_subscripts:
            max_used = max(used_subscripts)
            if max_used < 0:
                return 0
            else:
                return max_used + 1
        else:
            return 0

    used_keys: T.Dict[str, T.Set[T.Optional[int]]] = {}
    if excluded_keys is not None:
        for key in excluded_keys:
            mark_used(key)

    generated_keys = {}

    for name in names:
        letters_to_try, sub = _choices_for_name(name)
        generated_keys[name] = _pick_key_for_choices(
            letters_to_try, sub, is_unused, next_unused_sub
        )
        mark_used(generated_keys[name])

    return generated_keys
