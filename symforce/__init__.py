# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
Initialization configuration for symforce, as minimal as possible.
"""
from __future__ import absolute_import

from types import ModuleType
import typing as T
import os
import sys
import warnings

# -------------------------------------------------------------------------------------------------
# Version
# -------------------------------------------------------------------------------------------------

from ._version import version as __version__


# -------------------------------------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------------------------------------

import logging

# Create a logger with this print format
LOGGING_FORMAT = "%(module)s.%(funcName)s():%(lineno)s %(levelname)s -- %(message)s"
logging.basicConfig(format=LOGGING_FORMAT)
logger = logging.getLogger(__package__)


def set_log_level(log_level: str) -> None:
    """
    Set symforce logger level.

    The default is INFO, but can be set by one of:

        1) The SYMFORCE_LOGLEVEL environment variable
        2) Calling this function before any other symforce imports

    Args:
        log_level (str): {DEBUG, INFO, WARNING, ERROR, CRITICAL}
    """
    # Set default log level
    if not hasattr(logging, log_level.upper()):
        raise RuntimeError(f'Unknown log level: "{log_level}"')
    logger.setLevel(getattr(logging, log_level.upper()))

    # Only do this if already imported, in case users don't want to use any C++ binaries
    if "cc_sym" in sys.modules:
        import cc_sym

        cc_sym.set_log_level(log_level)


# Set default
set_log_level(os.environ.get("SYMFORCE_LOGLEVEL", "WARNING"))

# -------------------------------------------------------------------------------------------------
# Symbolic API configuration
# -------------------------------------------------------------------------------------------------


class InvalidSymbolicApiError(Exception):
    def __init__(self, api: str):
        super().__init__(f'Symbolic API is "{api}", must be one of ("sympy", "symengine")')


def _find_symengine() -> ModuleType:
    """
    Attempts to import symengine from its location in the symforce build directory

    If symengine is already in sys.modules, will return that module.  If symengine cannot be
    imported, raises ImportError.

    Returns the imported symengine module
    """
    if "symengine" in sys.modules:
        return sys.modules["symengine"]

    try:
        # If symengine is available on python path, use it
        # TODO(will, aaron): this might not be the version of symengine that we want
        import symengine

        return symengine
    except ImportError:
        pass

    import importlib
    import importlib.abc
    import importlib.util

    from . import path_util

    try:
        symengine_install_dir = path_util.symenginepy_install_dir()
    except path_util.MissingManifestException as ex:
        raise ImportError from ex

    symengine_path_candidates = list(
        symengine_install_dir.glob("lib/python3*/site-packages/symengine/__init__.py")
    ) + list(symengine_install_dir.glob("local/lib/python3*/dist-packages/symengine/__init__.py"))
    if len(symengine_path_candidates) != 1:
        raise ImportError(
            f"Should be exactly one symengine package, found candidates {symengine_path_candidates} in directory {path_util.symenginepy_install_dir()}"
        )
    symengine_path = symengine_path_candidates[0]

    # Import symengine from the directory where we installed it.  See
    # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    spec = importlib.util.spec_from_file_location("symengine", symengine_path)
    assert spec is not None
    symengine = importlib.util.module_from_spec(spec)
    sys.modules["symengine"] = symengine

    # For mypy: https://github.com/python/typeshed/issues/2793
    assert isinstance(spec.loader, importlib.abc.Loader)

    spec.loader.exec_module(symengine)

    return symengine


_symbolic_api: T.Optional[str] = None
_have_imported_symbolic = False


def _set_symbolic_api(sympy_module: str) -> None:
    # Set this as the default symbolic API
    global _symbolic_api  # pylint: disable=global-statement
    _symbolic_api = sympy_module


def _use_symengine() -> None:
    try:
        _find_symengine()

    except ImportError:
        logger.critical("Commanded to use symengine, but failed to import.")
        raise

    _set_symbolic_api("symengine")


def _use_sympy() -> None:
    # Import just to make sure it's importable and fail here if it's not (as opposed to failing
    # later)
    import sympy as sympy_py  # pylint: disable=unused-import

    _set_symbolic_api("sympy")


def set_symbolic_api(name: str) -> None:
    """
    Set the symbolic API for symforce. The sympy API is the default and pure python,
    whereas the symengine API is C++ and requires building the symengine library. It can
    be 100-200 times faster for many operations, but is less fully featured.

    The default is symengine if available else sympy, but can be set by one of:

        1) The SYMFORCE_SYMBOLIC_API environment variable
        2) Calling this function before any other symforce imports

    Args:
        name (str): {sympy, symengine}
    """
    if _have_imported_symbolic and name != _symbolic_api:
        raise ValueError(
            "The symbolic API cannot be changed after `symforce.symbolic` has been imported.  "
            "Import the top-level `symforce` module and call `symforce.set_symbolic_api` before "
            "importing anything else!"
        )

    if _symbolic_api is not None and name == _symbolic_api:
        logger.debug(f'already on symbolic API "{name}"')
        return
    else:
        logger.debug(f'symbolic API: "{name}"')

    if name == "sympy":
        _use_sympy()
    elif name == "symengine":
        _use_symengine()
    else:
        raise NotImplementedError(f'Unknown symbolic API: "{name}"')


# Set default to symengine if available, else sympy
if "SYMFORCE_SYMBOLIC_API" in os.environ:
    set_symbolic_api(os.environ["SYMFORCE_SYMBOLIC_API"])
else:
    try:
        _find_symengine()

        logger.debug("No SYMFORCE_SYMBOLIC_API set, found and using symengine.")
        set_symbolic_api("symengine")
    except ImportError:
        logger.debug("No SYMFORCE_SYMBOLIC_API set, no symengine found, using sympy.")
        set_symbolic_api("sympy")


def get_symbolic_api() -> str:
    """
    Return the current symbolic API as a string.

    Returns:
        str:
    """
    assert _symbolic_api is not None
    return _symbolic_api


# NOTE(hayk): Remove this after they are present in a release or two.


def get_backend() -> str:
    warnings.warn("`get_backend` is deprecated, use `get_symbolic_api`", FutureWarning)
    return get_symbolic_api()


def set_backend(name: str) -> None:
    warnings.warn("`set_backend` is deprecated use `set_symbolic_api`", FutureWarning)
    return set_symbolic_api(name)


# --------------------------------------------------------------------------------
# Default epsilon
# --------------------------------------------------------------------------------

# Should match C++ default epsilon in epsilon.h
numeric_epsilon = 10 * sys.float_info.epsilon


class AlreadyUsedEpsilon(Exception):
    """
    Exception thrown on attempting to modify the default epsilon after it has been used elsewhere
    """

    pass


_epsilon = 0.0
_have_used_epsilon = False


def _set_epsilon(new_epsilon: T.Any) -> None:
    """
    Set the default epsilon for SymForce

    This must be called before `symforce.symbolic` or other symbolic libraries have been imported.
    Typically it should be set to some kind of Scalar, such as an int, float, or Symbol.  See
    `symforce.symbolic.epsilon` for more information.

    Args:
        new_epsilon: The new default epsilon to use
    """
    if _have_used_epsilon:
        raise AlreadyUsedEpsilon(
            "Cannot set return value of epsilon after it has already been called."
        )

    global _epsilon  # pylint: disable=global-statement
    _epsilon = new_epsilon


def set_epsilon_to_symbol(name: str = "epsilon") -> None:
    """
    Set the default epsilon for Symforce to a Symbol.

    This must be called before `symforce.symbolic` or other symbolic libraries have been imported.
    See `symforce.symbolic.epsilon` for more information.

    Args:
        name: The name of the symbol for the new default epsilon to use
    """
    if get_symbolic_api() == "sympy":
        import sympy
    elif get_symbolic_api() == "symengine":
        sympy = _find_symengine()
    else:
        raise InvalidSymbolicApiError(get_symbolic_api())

    _set_epsilon(sympy.Symbol(name))


def set_epsilon_to_number(value: T.Any = numeric_epsilon) -> None:
    """
    Set the default epsilon for Symforce to a number.

    This must be called before `symforce.symbolic` or other symbolic libraries have been imported.
    See `symforce.symbolic.epsilon` for more information.

    Args:
        value: The new default epsilon to use
    """
    _set_epsilon(value)


def set_epsilon_to_zero() -> None:
    """
    Set the default epsilon for Symforce to zero.

    This must be called before `symforce.symbolic` or other symbolic libraries have been imported.
    See `symforce.symbolic.epsilon` for more information.
    """
    _set_epsilon(0.0)
