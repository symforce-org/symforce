# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import functools

from symforce import codegen
from symforce import typing as T
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
        :meth:`Codegen.lambdify <symforce.codegen.codegen.Codegen.lambdify>`
        :meth:`Codegen.function <symforce.codegen.codegen.Codegen.function>`
        :class:`codegen.PythonConfig <symforce.codegen.backends.python.python_config.PythonConfig>`
    """
    codegen_obj = codegen.Codegen.function(f, config=codegen.PythonConfig(use_numba=use_numba))
    return codegen_obj.lambdify()


def numbify(f: T.Callable) -> T.Callable:
    """
    Shorthand for ``lambdify(f, use_numba=True)``

    See also:
        :func:`lambdify`
    """
    return lambdify(f, use_numba=True)


SymbolicFunction = T.TypeVar("SymbolicFunction", bound=T.Callable)


def specialize_types(
    f: SymbolicFunction, type_replacements: T.Mapping[T.Type, T.Type]
) -> SymbolicFunction:
    """
    Specialize the type annotations on the given function, replacing any types in
    ``type_replacements``

    For example, this can be used to take a symbolic function that accepts a generic type and
    generate it for several concrete types::

        def f(x: sf.CameraCal) -> sf.Scalar:
            ...

        Codegen.function(specialize_types(f, {sf.CameraCal: sf.LinearCameraCal}), ...)
        Codegen.function(specialize_types(f, {sf.CameraCal: sf.PolynomialCameraCal}), ...)

    See also:
        :func:`specialize_args`
    """

    @functools.wraps(f)
    def specialized_function(*args: T.Any, **kwargs: T.Any) -> T.Any:
        return f(*args, **kwargs)

    specialized_function.__annotations__ = f.__annotations__.copy()

    for annotation, cls in specialized_function.__annotations__.items():
        if cls in type_replacements:
            specialized_function.__annotations__[annotation] = type_replacements[cls]

    return T.cast(SymbolicFunction, specialized_function)


def specialize_args(
    f: SymbolicFunction, arg_replacements: T.Mapping[str, T.Type]
) -> SymbolicFunction:
    """
    Specialize the type annotations on the given function, replacing types for any arguments in
    ``arg_replacements``

    For example, this can be used to take a symbolic function that accepts a generic type and
    generate it for several concrete types::

        def f(x: sf.CameraCal, y: sf.CameraCal) -> sf.Scalar:
            ...

        Codegen.function(
            specialize_types(f, {"x": sf.LinearCameraCal, "y": sf.PolynomialCameraCal}), ...
        )

    See also:
        :func:`specialize_types`
    """

    @functools.wraps(f)
    def specialized_function(*args: T.Any, **kwargs: T.Any) -> T.Any:
        return f(*args, **kwargs)

    specialized_function.__annotations__ = f.__annotations__.copy()

    for annotation in specialized_function.__annotations__:
        if annotation in arg_replacements:
            specialized_function.__annotations__[annotation] = arg_replacements[annotation]

    return T.cast(SymbolicFunction, specialized_function)
