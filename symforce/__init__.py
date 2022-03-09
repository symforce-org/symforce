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


# Set default
set_log_level(os.environ.get("SYMFORCE_LOGLEVEL", "WARNING"))

# -------------------------------------------------------------------------------------------------
# Symbolic backend configuration
# -------------------------------------------------------------------------------------------------
sympy: T.Any = None

from . import initialization


def _set_backend(sympy_module: ModuleType) -> None:
    # Make symforce-specific modifications to the sympy API
    initialization.modify_symbolic_api(sympy_module)

    # Set this as the default backend
    global sympy  # pylint: disable=global-statement
    sympy = sympy_module


def _import_symengine_from_build() -> ModuleType:
    """
    Attempts to import symengine from its location in the symforce build directory

    If symengine is already in sys.modules, will return that module.  If symengine cannot be
    imported, raises ImportError.

    Returns the imported symengine module
    """
    import sys

    if "symengine" in sys.modules:
        return sys.modules["symengine"]

    import importlib
    import importlib.abc
    import importlib.util

    from . import path_util

    symengine_path_candidates = list(
        path_util.symenginepy_install_dir().glob("lib/python3*/site-packages/symengine/__init__.py")
    )
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


def _use_symengine() -> None:
    try:
        symengine = _import_symengine_from_build()

    except ImportError:
        logger.critical("Commanded to use symengine, but failed to import.")
        raise

    _set_backend(symengine)


def _use_sympy() -> None:
    import sympy as sympy_py

    _set_backend(sympy_py)
    sympy_py.init_printing()

    # Hack in some key derivatives that sympy doesn't do. For all these
    # cases the derivative is zero except at the discrete switching point,
    # and assuming zero is correct for our numerical purposes.
    setattr(sympy.floor, "_eval_derivative", lambda s, v: sympy.S.Zero)
    setattr(sympy.sign, "_eval_derivative", lambda s, v: sympy.S.Zero)
    setattr(sympy.Mod, "_eval_derivative", lambda s, v: sympy.S.Zero)


def set_backend(backend: str) -> None:
    """
    Set the symbolic backend for symforce. The sympy backend is the default and pure python,
    whereas the symengine backend is C++ and requires building the symengine library. It can
    be 100-200 times faster for many operations, but is less fully featured.

    The default is symengine if available else sympy, but can be set by one of:

        1) The SYMFORCE_BACKEND environment variable
        2) Calling this function before any other symforce imports

    Args:
        backend (str): {sympy, symengine}
    """
    # TODO(hayk): Could do a better job of checking what's imported and raising an error
    # if this isn't the first thing imported/called from symforce.

    if sympy and backend == sympy.__package__:
        logger.debug(f'already on backend "{backend}"')
        return
    else:
        logger.debug(f'backend: "{backend}"')

    if backend == "sympy":
        _use_sympy()
    elif backend == "symengine":
        _use_symengine()
    else:
        raise NotImplementedError(f'Unknown backend: "{backend}"')


# Set default to symengine if available, else sympy
if "SYMFORCE_BACKEND" in os.environ:
    set_backend(os.environ["SYMFORCE_BACKEND"])
else:
    try:
        symengine = _import_symengine_from_build()

        logger.debug("No SYMFORCE_BACKEND set, found and using symengine.")
        set_backend("symengine")
    except ImportError:
        logger.debug("No SYMFORCE_BACKEND set, no symengine found, using sympy.")
        set_backend("sympy")


def get_backend() -> str:
    """
    Return the current backend as a string.

    Returns:
        str:
    """
    assert sympy is not None
    return sympy.__name__
