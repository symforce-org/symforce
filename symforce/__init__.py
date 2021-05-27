"""
Initialization configuration for symforce, as minimal as possible.
"""
import typing as T
import os

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
set_log_level(os.environ.get("SYMFORCE_LOGLEVEL", "INFO"))

# -------------------------------------------------------------------------------------------------
# Symbolic backend configuration
# -------------------------------------------------------------------------------------------------
sympy: T.Any = None

from . import initialization


def _set_backend(sympy_module: T.Type) -> None:
    # Make symforce-specific modifications to the sympy API
    initialization.modify_symbolic_api(sympy_module)

    # Set this as the default backend
    global sympy  # pylint: disable=global-statement
    sympy = sympy_module


def _use_symengine() -> None:
    try:
        import symengine
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
        import symengine

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
