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
from .impl.sequence_lie_group_ops import SequenceLieGroupOps

LieGroupOps.register(float, ScalarLieGroupOps)
LieGroupOps.register(np.float32, ScalarLieGroupOps)
LieGroupOps.register(np.float64, ScalarLieGroupOps)
LieGroupOps.register(sm.Expr, ScalarLieGroupOps)

# NOTE(hayk): It's weird to call integers lie groups, but the implementation of ScalarLieGroupOps
# converts everything to symbolic types so it acts like a floating point.
LieGroupOps.register(int, ScalarLieGroupOps)
LieGroupOps.register(np.int64, ScalarLieGroupOps)

LieGroupOps.register(list, SequenceLieGroupOps)
LieGroupOps.register(tuple, SequenceLieGroupOps)
# TODO(nathan): I think we need to write a new set of implementations for numpy arrays
# because they can be 2D; the current SequenceOps won't work properly for them (except for
# to_storage)
LieGroupOps.register(np.ndarray, SequenceLieGroupOps)
