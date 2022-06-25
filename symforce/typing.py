# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
Common type definitions.
"""
import abc
import dataclasses
import os

# Expose all types.
from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

# This is kind of a heavy/unnecessary dependency,here so only import when type checking so we can
# resolve the annotation below
if TYPE_CHECKING:
    import numpy as np

# TODO(hayk,aaron): Either make this a union of "Scalar types", or different typevars for numeric
# and symbolic scalars or something
if TYPE_CHECKING:
    # Currently this can be any type, and doesn't even express that multiple Scalars in a signature
    # are the same (which is usually or always the case).  However, making this a TypeVar with
    # a loose enough bound is similarly annoying
    Scalar = Any
else:
    Scalar = float

# Alias for argument type of open, which typing does not seem to have.  We don't include int because
# who uses that anyway, and bytes because some things in os.path don't support that
Openable = Union[str, os.PathLike]

# Represents any Group element object
Element = Any

# Represents any Group element type or object
ElementOrType = Union[Element, Type]

# Specialization for scalar elements
ScalarElement = Scalar
ScalarElementOrType = Union[ScalarElement, Type[ScalarElement]]

# Specialization for sequence elements
SequenceElement = Sequence[Element]
SequenceElementOrType = Union[SequenceElement, Type[SequenceElement]]

# Specialization for array elements
# We need "Union" here to avoid import errors associated with numpy only being imported when type
# checking. Without "Union" mypy thinks our type alias is just a string, not a type alias.
# See https://mypy.readthedocs.io/en/stable/kinds_of_types.html#type-aliases
# This could be improved after we upgrade to Mypy 0.930 or later by using "TypeAlias"
ArrayElement = Union["np.ndarray"]
ArrayElementOrType = Union[ArrayElement, Type[ArrayElement]]

# Dataclass Metaclass
if TYPE_CHECKING:
    # Mypy doesn't understand __subclasshook__
    Dataclass = Any
else:

    class Dataclass(abc.ABC):
        """
        Metaclass for dataclasses (which do not have a common superclass)
        """

        @abc.abstractmethod
        def __init__(self, *args: Any, **kwargs: Any):
            pass

        @classmethod
        def __subclasshook__(cls, subclass: Type) -> bool:
            return dataclasses.is_dataclass(subclass) and isinstance(subclass, type)


DataclassOrType = Union[Dataclass, Type[Dataclass]]

# Abstract method helpers
_ReturnType = TypeVar("_ReturnType")


def any_args(f: Callable[..., _ReturnType]) -> Callable[..., _ReturnType]:
    """
    Decorator to mark an abstract method as allowing subclasses to override with any argument types.

    THIS LIES TO THE TYPE CHECKER, AND ALLOWS VIOLATION OF THE LISKOV SUBSTITUTION PRINCIPLE. USE
    ONLY ON FUNCTIONS THAT WILL NEVER BE CALLED IN A CONTEXT THAT KNOWS ONLY THE BASE TYPE.
    """
    return f
