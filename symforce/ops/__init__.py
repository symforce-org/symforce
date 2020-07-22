"""
API for mathematical groups in python with minimal dependencies. Assumes elements
have appropriate methods, or for the case of scalar types (ints, floats, sympy.Symbols)
assumes that the group is reals under addition.

This is the recommended API for using these concepts, rather than calling directly on a type.
"""

from .storage_ops import StorageOps
from .group_ops import GroupOps
from .lie_group_ops import LieGroupOps

# Register ops for scalars and sequences
import numpy as np
from symforce import sympy as sm
from .impl.scalar_storage_ops import ScalarStorageOps
from .impl.scalar_lie_group_ops import ScalarLieGroupOps
from .impl.sequence_storage_ops import SequenceStorageOps

# TODO(nathan): Not sure if we actually want to support ints. Currently there's a subtle
# bug where you can perturb an integer by a scalar in the tangent space, but when we call
# "from_tangent" the decimal part will be dropped.
StorageOps.register(int, ScalarStorageOps)
LieGroupOps.register(float, ScalarLieGroupOps)
LieGroupOps.register(np.float32, ScalarLieGroupOps)
LieGroupOps.register(np.float64, ScalarLieGroupOps)
LieGroupOps.register(sm.Expr, ScalarLieGroupOps)

StorageOps.register(list, SequenceStorageOps)
StorageOps.register(tuple, SequenceStorageOps)
StorageOps.register(np.ndarray, SequenceStorageOps)
