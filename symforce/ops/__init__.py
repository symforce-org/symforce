"""
API for mathematical groups in python with minimal dependencies. Assumes elements
have appropriate methods, or for the case of scalar types (ints, floats, sympy.Symbols)
assumes that the group is reals under addition.

This is the recommended API for using these concepts, rather than calling directly on a type.
"""

from .storage_ops import StorageOps
from .group_ops import GroupOps
from .lie_group_ops import LieGroupOps
