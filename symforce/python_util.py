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


def remove_if_exists(path):
    # type: (str) -> None
    """
    Delete a file or directory if it exists.
    """
    if not os.path.exists(path):
        logger.debug("Doesn't exist: {}".format(path))
        return
    elif os.path.isdir(path):
        logger.debug("Removing directory: {}".format(path))
        shutil.rmtree(path)
    else:
        logger.debug("Removing file: {}".format(path))
        os.remove(path)


def execute_subprocess(
    cmd,  # type: T.Union[str, T.Sequence[str]]
    *args,  # type: T.Any
    **kwargs  # type: T.Any
):
    # type: (...) -> None
    """
    Execute subprocess and log command as well as stdout/stderr.

    Raises:
        subprocess.CalledProcessError: If the return code is nonzero
    """
    cmd_str = " ".join(cmd) if isinstance(cmd, (tuple, list)) else cmd
    logger.info("Subprocess: {}".format(cmd_str))

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, *args, **kwargs)  # type: ignore
    (stdout, _) = proc.communicate()
    logger.info(stdout.decode("utf-8"))

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, stdout)


def camelcase_to_snakecase(s):
    # type: (str) -> str
    """
    Convert CamelCase -> snake_case.
    """
    return re.sub(r"(?<!^)(?=[A-Z][a-z])", "_", s).lower()


def files_in_dir(dirname, relative=False):
    # type: (str, bool) -> T.Iterator[str]
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


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    # type: (int, str) -> str
    """
    Generate a random string within a character set - for example "6U1S75".
    This is not cryptographically secure.
    """
    return "".join(random.choice(chars) for _ in range(size))


def get_type(a):
    # type: (T.Any) -> T.Type
    """
    Returns the type of the element if its an instance, or a pass through if already a type.
    """
    if isinstance(a, type):
        return a
    else:
        return type(a)


def scalar_like(a):
    # type: (T.Any) -> bool
    """
    Returns whether the element is scalar-like (an int, float, or sympy expression).

    This method does not rely on the value of a, only the type.
    """
    a_type = get_type(a)
    if issubclass(a_type, (int, float, np.float32, np.float64)):
        return True
    is_expr = issubclass(a_type, sm.Expr)
    is_matrix = issubclass(a_type, sm.MatrixBase)
    return is_expr and not is_matrix
