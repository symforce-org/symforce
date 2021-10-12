"""
Common type definitions.
"""
import abc
import dataclasses
import os

# Expose all types.
from typing import *

# This is kind of a heavy/unnecessary dependency,here so only import when type checking so we can
# resolve the annotation below
if TYPE_CHECKING:
    import numpy as np

# TODO(hayk): Try to make a basic type stub for sympy Expr.
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
ScalarElementOrType = Union[ScalarElement, Type]

# Specialization for sequence elements
SequenceElement = Union[Sequence[Element], "np.ndarray"]
SequenceElementOrType = Union[SequenceElement, Type]


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


# Abstract method helpers
_ReturnType = TypeVar("_ReturnType")


def any_args(f: Callable[..., _ReturnType]) -> Callable[..., _ReturnType]:
    """
    Decorator to mark an abstract method as allowing subclasses to override with any argument types.

    THIS LIES TO THE TYPE CHECKER, AND ALLOWS VIOLATION OF THE LISKOV SUBSTITUTION PRINCIPLE. USE
    ONLY ON FUNCTIONS THAT WILL NEVER BE CALLED IN A CONTEXT THAT KNOWS ONLY THE BASE TYPE.
    """
    return f
