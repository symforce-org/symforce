import builtins
import inspect

from symforce import cam
from symforce import geo
from symforce import types as T


def deduce_input_type(
    parameter: inspect.Parameter, func: T.Callable, is_first_parameter: bool
) -> T.ElementOrType:
    """
    Attempt to deduce the type of an input parameter to a function

    Strategy:
    1) If it's annotated with something that's not a string, return that
    2) If it's the first parameter and its name is "self", search for a type by the class part of
        the function's qualified name
    3) If the annotation is a string:
        3a) If it's one piece (i.e. no dot), look for that type in builtins, T, geo, and cam
        3b) If it's two pieces, the first has to be T, geo, or cam, we look for the second piece
            in those modules
        3c) We don't support more than 2 pieces, i.e. names like geo.matrix.Vector3 aren't
            supported (use geo.Vector3 if possible)
    """
    annotation = parameter.annotation

    # 1)
    if annotation is not parameter.empty and not isinstance(annotation, str):
        return annotation

    # 2)
    if annotation is parameter.empty:
        if is_first_parameter and parameter.name == "self":
            # self is unannotated, so try setting the annotation to the class name
            # __qualname__ should be of the form Class.func_name
            annotation = func.__qualname__.rsplit(".", maxsplit=1)[0]

            # Try to fetch the type from geo
            try:
                return getattr(geo, annotation)
            except AttributeError as ex:
                raise ValueError(
                    f"Type for argument self to {func} could not be deduced."
                    + "  Please annotate the `self` parameter on your function,"
                    + " or provide input_types"
                ) from ex
        else:
            raise ValueError(
                f'Type for argument "{parameter.name}" to {func} could not be deduced.'
                + "  Please either provide input_types or add a type annotation"
            )

        assert False, "All paths through this block should return or raise"

    # 3)
    assert isinstance(annotation, str), "If we reach this point annotation should be a str"
    annotation_parts = annotation.split(".")

    error_msg = (
        f'Type annotation for argument "{parameter.name}" to {func} is the string '
        + f'"{annotation}", and could not be resolved to the actual type'
    )

    extra_modules_to_try = [T, geo, cam]

    if len(annotation_parts) == 1:
        # 3a)
        all_modules_to_try = [builtins] + extra_modules_to_try
        for module in all_modules_to_try:
            try:
                annotation = getattr(module, annotation)
            except (AttributeError, KeyError) as ex:
                pass
            else:
                return annotation

        all_modules_str = ", ".join([str(m.__name__) for m in all_modules_to_try])
        raise ValueError(
            f"{error_msg}; tried looking in {all_modules_str}.  Please provide input_types"
        )
    elif len(annotation_parts) == 2:
        # 3b)

        # Include "T" as well, because it's resolved as types
        extra_modules_strs = ["T"] + [str(m.__name__.split(".")[-1]) for m in extra_modules_to_try]

        if annotation_parts[0] not in extra_modules_strs:
            raise ValueError(
                f"{error_msg} because it is a multi-part name and the first part is"
                + f" {annotation_parts[0]}, not {', '.join(extra_modules_strs)}"
            )

        try:
            return getattr(globals()[annotation_parts[0]], annotation_parts[1])
        except AttributeError as ex:
            raise ValueError(
                f"{error_msg} because that type does not exist in {annotation_parts[0]}"
            ) from ex
    else:
        # 3c)
        raise ValueError(
            f"{error_msg} because it has {len(annotation_parts)} nested names"
            + " (can only have 1 or 2)"
        )

    assert False, "All paths should return or raise before this point"


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
