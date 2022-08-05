# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
General python utilities.
"""
import functools
import inspect
import os
import random
import re
import shutil
import string
import subprocess

from symforce import logger
from symforce import typing as T


def remove_if_exists(path: T.Openable) -> None:
    """
    Delete a file or directory if it exists.
    """
    if not os.path.exists(path):
        logger.debug(f"Doesn't exist: {path}")
        return
    elif os.path.isdir(path):
        logger.debug(f"Removing directory: {path}")
        shutil.rmtree(path)
    else:
        logger.debug(f"Removing file: {path}")
        os.remove(path)


def execute_subprocess(
    cmd: T.Union[str, T.Sequence[str]],
    stdin_data: T.Optional[str] = None,
    log_stdout: bool = True,
    log_stdout_to_error_on_error: bool = True,
    **kwargs: T.Any,
) -> str:
    """
    Execute subprocess and log command as well as stdout/stderr.

    Args:
        stdin_data (bytes): Data to pass to stdin
        log_stdout (bool): Write process stdout to the logger?
        log_stdout_to_error_on_error: Write output to logger.error if the command fails?

    Raises:
        subprocess.CalledProcessError: If the return code is nonzero
    """
    if stdin_data is not None:
        stdin_data_encoded = stdin_data.encode("utf-8")
    else:
        stdin_data_encoded = bytes()

    cmd_str = " ".join(cmd) if isinstance(cmd, (tuple, list)) else cmd
    logger.info(f"Subprocess: {cmd_str}")

    with subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs
    ) as proc:
        (stdout, _) = proc.communicate(stdin_data_encoded)
        return_code = proc.returncode

    going_to_log_to_err = return_code != 0 and log_stdout_to_error_on_error

    stdout_decoded = stdout.decode("utf-8")
    if log_stdout and not going_to_log_to_err:
        logger.info(stdout_decoded)

    if going_to_log_to_err:
        logger.error(
            f"Subprocess {cmd} exited with code: {return_code}.  Output:\n{stdout_decoded}"
        )

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd, stdout_decoded)

    return stdout_decoded


def camelcase_to_snakecase(s: str) -> str:
    """
    Convert CamelCase -> snake_case.
    """
    return re.sub(r"(?<!^)(?=[A-Z][a-z])", "_", s).lower()


def snakecase_to_camelcase(s: str) -> str:
    """
    Convert snake_case -> CamelCase

    Double underscores are escaped, e.g. one__two becomes One_Two
    """
    return (
        re.sub("_+", lambda match: "_" * (len(match.group()) // 2) + " ", s)
        .title()
        .replace(" ", "")
    )


def camelcase_to_screaming_snakecase(s: str) -> str:
    """
    Convert CamelCase -> SCREAMING_SNAKE_CASE
    """
    return camelcase_to_snakecase(s).upper()


def str_replace_all(s: str, replacements: T.Dict[str, str]) -> str:
    """
    Call str.replace(old, new) for every pair (old, new) in replacements
    """
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def str_removeprefix(s: str, prefix: str) -> str:
    """
    Backport of str.removeprefix, from Python3.9
    https://docs.python.org/3/library/stdtypes.html#str.removeprefix

    If the string starts with the prefix string and that prefix is not empty, return
    string[len(prefix):]. Otherwise, return a copy of the original string.
    """
    if s.startswith(prefix):
        return s[len(prefix) :]
    else:
        return s


def str_removesuffix(s: str, suffix: str) -> str:
    """
    Backport of str.removesuffix, from Python3.9
    https://docs.python.org/3/library/stdtypes.html#str.removesuffix

    If the string ends with the suffix string and that suffix is not empty, return
    string[:-len(suffix)]. Otherwise, return a copy of the original string.
    """
    if s.endswith(suffix):
        return s[: -len(suffix)]
    else:
        return s


def files_in_dir(dirname: T.Openable, relative: bool = False) -> T.Iterator[str]:
    """
    Return a list of files in the given directory.
    """
    for root, _, filenames in os.walk(os.fspath(dirname)):
        for filename in filenames:
            abspath = os.path.join(root, filename)
            if relative:
                yield os.path.relpath(abspath, dirname)
            else:
                yield abspath


def id_generator(size: int = 6, chars: str = string.ascii_uppercase + string.digits) -> str:
    """
    Generate a random string within a character set - for example "6U1S75".
    This is not cryptographically secure.
    """
    return "".join(random.choice(chars) for _ in range(size))


def getattr_recursive(obj: object, attrs: T.Sequence[str]) -> T.Any:
    """
    Recursively calls getattr on obj with the attributes in attrs and returns the output.
    If attr is empty, returns obj.

    Example:
        get_attr_recursive(obj, ["A", "B", "C"]) returns the same thing as
        obj.A.B.C
    """
    return getattr_recursive(getattr(obj, attrs[0]), attrs[1:]) if len(attrs) else obj


class InvalidKeyError(ValueError):
    pass


class InvalidPythonIdentifierError(InvalidKeyError):
    pass


def base_and_indices(indexed_array: str) -> T.Tuple[str, T.List[int]]:
    r"""
    Decomposes indexed_array into (base, indices) in the sense that,
    "arr[1][2]" -> ("arr", [1, 2]). base is the initial substring of indexed_array
    that does not contain either "[" or "]"; indices is is the list of integer indices
    indexing into the array denoted by base.

    indices will be the empty list if indexed_array has no indices.

    Raises:
        InvalidKeyError if indexed_array is not matched by the regular expression
        r"[\[\]]*(\[[0-9]+\])*", i.e., is not a string with no square brackets,
        followed by 0 or more integers wrapped in square brackets.

    Example:
        >>> assert ("arr", []) == base_and_indices("arr")
        >>> assert ("arr", [1, 2, 3]) == base_and_indices("arr[1][2][3]")
        >>> try:
        >>>     base_and_indices("arr[1].bad[2]")
        >>>     assert False
        >>> except ValueError:
        >>>     pass
        >>> except:
        >>>     assert False

    """
    base_indices_match = re.fullmatch(r"([^\[\]]*)((?:\[[0-9]+\])*)", indexed_array)
    if base_indices_match is None:
        raise InvalidKeyError(f"{indexed_array} is not a base and its indices")
    base, indices_str = base_indices_match.groups()
    indices = [int(match.group()) for match in re.finditer(r"[0-9]+", indices_str)]
    return base, indices


def plural(singular: str, count: int, plural: str = None) -> str:
    """
    Return the singular or plural form of a word based on count

    Adds an s to singular by default for the plural form, or uses `plural` if provided
    """
    if count == 1:
        return singular
    else:
        return plural or (singular + "s")


def get_func_from_maybe_bound_function(func: T.Callable) -> T.Callable:
    """
    Get the original function, from a function possibly bound by functools.partial
    """
    if isinstance(func, functools.partial):
        return func.func
    else:
        return func


def get_class_for_method(func: T.Callable) -> T.Type:
    """
    Get the class from an instance method `func`

    See https://stackoverflow.com/a/25959545
    """
    if isinstance(func, functools.partial):
        return get_class_for_method(func.func)
    if inspect.ismethod(func) or (
        inspect.isbuiltin(func)
        and getattr(func, "__self__", None) is not None
        and getattr(getattr(func, "__self__"), "__class__", None) is not None
    ):
        return getattr(getattr(func, "__self__"), "__class__")
    if inspect.isfunction(func):
        cls = getattr(
            inspect.getmodule(func),
            func.__qualname__.split(".<locals>", 1)[0].rsplit(".", 1)[0],
            None,
        )
        if isinstance(cls, type):
            return cls
    return getattr(func, "__objclass__")  # handle special descriptor objects


class AttrDict(dict):
    """
    A simple attr-dict, i.e. a dictinary whose keys are also accessible directly as fields

    Based on http://stackoverflow.com/a/14620633/53997
    """

    def __init__(self, **kwargs: T.Any) -> None:
        super().__init__(**kwargs)
        self.__dict__ = self

    if T.TYPE_CHECKING:
        # The existence of these methods makes mypy happy with accessing members of this object,
        # since it isn't quite able to figure out how __dict__ works.
        # Note that this code has no effect at runtime because T.TYPE_CHECKING evaluates to False,
        # so this code never runs and these methods are not defined.
        def __getattr__(self, name: str) -> T.Any:
            pass

        def __setattr__(self, name: str, value: T.Any) -> None:
            pass
