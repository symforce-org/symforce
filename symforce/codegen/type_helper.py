import inspect

from symforce import typing as T
from symforce import python_util


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
        return python_util.get_type_hints_of_maybe_bound_function(func)[parameter.name]

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
    for i, (parameter_name, parameter) in enumerate(signature.parameters.items()):
        input_types.append(deduce_input_type(parameter, func, i == 0))

    return input_types
