# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from symforce import codegen
from symforce import typing as T
from symforce.codegen import codegen_util
from symforce.type_helpers import symbolic_inputs

_T = T.TypeVar("_T")


def symbolic_eval(func: T.Callable[..., _T]) -> _T:
    """
    Build symbolic arguments for a function, and return the function evaluated on those arguments.

    Useful for easily visualizing what expressions a symbolic function produces

    Args:
        func: A callable; args should have type annotations, and those types should be constructible
              automatically with :func:`symforce.ops.storage_ops.StorageOps.symbolic`

    Returns:
        The outputs of ``func`` evaluated on the constructed symbolic args

    See also:
        :func:`symforce.type_helpers.symbolic_inputs`
    """
    return func(**symbolic_inputs(func))


def lambdify(f: T.Callable, use_numba: bool = False) -> T.Callable:
    """
    Convert a symbolic function to a numerical one.  This is a thin wrapper around
    :meth:`Codegen.function <symforce.codegen.codegen.Codegen.function>` provided for convenience.

    Args:
        f: A callable with symbolic inputs and outputs - see
            :meth:`Codegen.function <symforce.codegen.codegen.Codegen.function>` for details
        use_numba: If True, use Numba to compile the generated function.  This can be much faster,
                   but has some limitations - see
                   :class:`codegen.PythonConfig <symforce.codegen.backends.python.python_config.PythonConfig>`
                   for details

    Returns:
        A numerical function equivalent to ``f``

    See also:
        :meth:`Codegen.function <symforce.codegen.codegen.Codegen.function>`
        :class:`codegen.PythonConfig <symforce.codegen.backends.python.python_config.PythonConfig>`
    """
    codegen_obj = codegen.Codegen.function(f, config=codegen.PythonConfig(use_numba=use_numba))
    data = codegen_obj.generate_function(namespace=f.__name__)
    assert codegen_obj.name is not None
    generated_function = codegen_util.load_generated_function(
        codegen_obj.name, data.function_dir, evict=not use_numba
    )
    return generated_function


def numbify(f: T.Callable) -> T.Callable:
    """
    Shorthand for ``lambdify(f, use_numba=True)``

    See also:
        :func:`lambdify`
    """
    return lambdify(f, use_numba=True)
