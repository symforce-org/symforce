"""
General python utilities.
"""
import os
import random
import re
import shutil
import string
import subprocess
import numpy as np

from symforce import logger
from symforce import sympy as sm
from symforce import types as T


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

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)  # type: ignore
    (stdout, _) = proc.communicate(stdin_data_encoded)

    going_to_log_to_err = proc.returncode != 0 and log_stdout_to_error_on_error

    stdout_decoded = stdout.decode("utf-8")
    if log_stdout and not going_to_log_to_err:
        logger.info(stdout_decoded)

    if going_to_log_to_err:
        logger.error(
            f"Subprocess {cmd} exited with code: {proc.returncode}.  Output:\n{stdout_decoded}"
        )

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, stdout_decoded)

    return stdout_decoded


def camelcase_to_snakecase(s: str) -> str:
    """
    Convert CamelCase -> snake_case.
    """
    return re.sub(r"(?<!^)(?=[A-Z][a-z])", "_", s).lower()


def snakecase_to_camelcase(s: str) -> str:
    """
    Convert snake_case -> CamelCase
    """
    return s.replace("_", " ").title().replace(" ", "")


def str_replace_all(s: str, replacements: T.Dict[str, str]) -> str:
    """
    Call str.replace(old, new) for every pair (old, new) in replacements
    """
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def files_in_dir(dirname: T.Openable, relative: bool = False) -> T.Iterator[str]:
    """
    Return a list of files in the given directory.
    """
    for root, _, filenames in os.walk(dirname):
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


def get_type(a: T.Any) -> T.Type:
    """
    Returns the type of the element if its an instance, or a pass through if already a type.
    """
    if isinstance(a, type):
        return a
    else:
        return type(a)


def scalar_like(a: T.Any) -> bool:
    """
    Returns whether the element is scalar-like (an int, float, or sympy expression).

    This method does not rely on the value of a, only the type.
    """
    a_type = get_type(a)
    if issubclass(a_type, (int, float, np.float32, np.float64)):
        return True
    is_expr = issubclass(a_type, sm.Expr)
    is_matrix = issubclass(a_type, sm.MatrixBase) or (hasattr(a, "is_Matrix") and a.is_Matrix)
    return is_expr and not is_matrix


def getattr_recursive(obj: object, attrs: T.Sequence[str]) -> T.Any:
    """
    Recursively calls getattr on obj with the attributes in attrs and returns the output.
    If attr is empty, returns obj.

    Example:
        get_attr_recursive(obj, ["A", "B", "C"]) returns the same thing as
        obj.A.B.C
    """
    return getattr_recursive(getattr(obj, attrs[0]), attrs[1:]) if len(attrs) else obj


def base_and_indices(indexed_array: str) -> T.Tuple[str, T.List[int]]:
    """
    Decomposes indexed_array into (base, indices) in the sense that,
    "arr[1][2]" -> ("arr", [1, 2]). base is the initial substring of indexed_array
    that does not contain either "[" or "]"; indices is is the list of integer indices
    indexing into the array denoted by base.

    indices will be the empty list if indexed_array has no indices.

    Raises:
        ValueError if indexed_array is not matched by the regular expression
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
        raise ValueError(f"{indexed_array} is not a base and its indices")
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
