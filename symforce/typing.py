"""
Common type definitions.
"""
import os

# Expose all types.
from typing import *

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
SequenceElement = Sequence[Element]
SequenceElementOrType = Union[SequenceElement, Type]
