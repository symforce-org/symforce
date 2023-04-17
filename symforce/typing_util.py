# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
General python typing-related utilities
"""

import dataclasses

import numpy as np

import symforce.internal.symbolic as sf
from symforce import typing as T


def get_type(a: T.Any) -> T.Type:
    """
    Returns the type of the element if its an instance, or a pass through if already a type.
    """
    if isinstance(a, type):
        return a
    else:
        return type(a)


# NOTE(brad): Each of these classes is automatically registered with ScalarLieGroupOps. (see
# ops/__init__.py)
SCALAR_TYPES = (
    float,
    np.float16,
    np.float32,
    np.float64,
    # NOTE(hayk): It's weird to call integers lie groups, but the implementation of ScalarLieGroupOps
    # converts everything to symbolic types so it acts like a floating point.
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
)


def scalar_like(a: T.Any) -> bool:
    """
    Returns whether the element is scalar-like (an int, float, or sympy expression).

    This method does not rely on the value of a, only the type.
    """
    a_type = get_type(a)
    if issubclass(a_type, SCALAR_TYPES):
        return True

    is_expr = issubclass(a_type, sf.Expr)
    if not is_expr:
        return False

    # It is an expr, check that it's not a matrix
    is_matrix = issubclass(a_type, sf.sympy.MatrixBase) or (hasattr(a, "is_Matrix") and a.is_Matrix)
    return not is_matrix


def get_sequence_from_dataclass_sequence_field(
    field: dataclasses.Field, field_type: T.Type
) -> T.Sequence[T.Any]:
    origin = T.get_origin(field_type)
    length = field.metadata.get("length")
    if origin is None or not issubclass(origin, T.Sequence):
        raise TypeError(
            f"Annotated field with `length={length}` that is of type {field_type}, not T.Sequence"
        )
    assert isinstance(length, int)
    arg_type = T.get_args(field_type)[0]
    return [arg_type] * length


def maybe_tuples_of_types_from_annotation(
    annotation: T.Union[T.Type, T.Any], return_annotation_if_not_tuple: bool = False
) -> T.Optional[T.Union[T.Tuple[T.Union[T.Tuple, T.Type]], T.Any]]:
    """
    Attempt to construct a tuple of types from an annotation of the form T.Tuple[A, B, C] of any
    fixed length, recursively.

    If this is not possible, because the annotation is not a T.Tuple, returns:

    1) The annotation itself, if return_annotation_if_not_tuple is True
    2) None, otherwise

    If the annotation is a T.Tuple, but is of unknown length, returns None
    """
    origin = T.get_origin(annotation)
    if not isinstance(origin, type) or not issubclass(origin, T.cast(T.Type, T.Tuple)):
        if return_annotation_if_not_tuple:
            return annotation
        else:
            return None

    args = T.get_args(annotation)

    if Ellipsis in args:
        if return_annotation_if_not_tuple:
            raise ValueError()
        else:
            return None

    return tuple(
        maybe_tuples_of_types_from_annotation(arg, return_annotation_if_not_tuple=True)
        for arg in args
    )
