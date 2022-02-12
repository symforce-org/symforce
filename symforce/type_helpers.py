# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import inspect

from symforce import ops
from symforce import python_util
from symforce import typing as T
from symforce.values import Values


def deduce_input_type(
    parameter: inspect.Parameter, func: T.Callable, is_first_parameter: bool
) -> T.ElementOrType:
    """
    Attempt to deduce the type of an input parameter to a function

    Strategy:
    1) If it's annotated with something, return that
    2) If it's the first parameter and its name is "self", search for a type by the class part of
        the function's qualified name
    """
    annotation = parameter.annotation

    # 1)
    if annotation is not parameter.empty:
        return T.get_type_hints(python_util.get_func_from_maybe_bound_function(func))[
            parameter.name
        ]

    # 2)
    if is_first_parameter and parameter.name == "self":
        # self is unannotated, so try setting the annotation to the class name
        # __qualname__ should be of the form Class.func_name
        return python_util.get_class_for_method(func)

    raise ValueError(
        f'Type for argument "{parameter.name}" to {func} could not be deduced.'
        + "  Please either provide input_types or add a type annotation"
    )


def deduce_input_types(func: T.Callable) -> T.Sequence[T.ElementOrType]:
    """
    Attempt to deduce input types from the type annotations on func, to be used by Codegen.function.

    See the docstring on deduce_input_type for deduction strategy
    """
    signature = inspect.signature(func)

    input_types = []
    for i, parameter in enumerate(signature.parameters.values()):
        input_types.append(deduce_input_type(parameter, func, i == 0))

    return input_types


def symbolic_inputs(func: T.Callable, input_types: T.Sequence[T.ElementOrType] = None) -> Values:
    """
    Return symbolic arguments for the inputs to `func`

    Args:
        func: A callable; args should have type annotations, and those types should be constructible
              automatically with StorageOps.symbolic

    Returns:
        A tuple with a symbolic object for each input to func
    """
    parameters = [
        p
        for p in inspect.signature(func).parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]

    if input_types is None:
        input_types = deduce_input_types(func)
    else:
        assert len(parameters) == len(
            input_types
        ), f"Parameters: {parameters}, inputs_types: {input_types}"

    # Formulate symbolic arguments to function
    inputs = Values()
    for arg_parameter, arg_type in zip(parameters, input_types):
        inputs[arg_parameter.name] = ops.StorageOps.symbolic(arg_type, arg_parameter.name)

    return inputs


_T = T.TypeVar("_T")


def symbolic_eval(func: T.Callable[..., _T]) -> _T:
    """
    Build symbolic arguments for a function, and return the function evaluated on those arguments.

    Useful for easily visualizing what expressions a symbolic function produces

    Args:
        func: A callable; args should have type annotations, and those types should be constructible
              automatically with StorageOps.symbolic

    Returns:
        The outputs of `func` evaluated on the constructed symbolic args
    """
    return func(**symbolic_inputs(func))
