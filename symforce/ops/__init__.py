# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

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
from .impl.scalar_lie_group_ops import ScalarLieGroupOps
from .impl.sequence_lie_group_ops import SequenceLieGroupOps
from .impl.array_storage_ops import ArrayStorageOps

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

StorageOps.register(np.ndarray, ArrayStorageOps)

from symforce import typing as T
from .impl.dataclass_storage_ops import DataclassStorageOps

StorageOps.register(T.Dataclass, DataclassStorageOps)

# TODO(hayk): Are these okay here or where can we put them? In theory we could just have this
# be automatic that if the given type has the methods that it gets registered automatically.
import sym
from .impl.class_lie_group_ops import ClassLieGroupOps

LieGroupOps.register(sym.Rot2, ClassLieGroupOps)
LieGroupOps.register(sym.Rot3, ClassLieGroupOps)
LieGroupOps.register(sym.Pose2, ClassLieGroupOps)
LieGroupOps.register(sym.Pose3, ClassLieGroupOps)
