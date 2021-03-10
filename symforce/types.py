"""
Common type definitions.
"""
# Expose all types.
from typing import *

# TODO(hayk): Try to make a basic type stub for sympy Expr.
Scalar = float

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
