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

import abc

import numpy as np

from .group_ops import GroupOps
from .lie_group_ops import LieGroupOps
from .storage_ops import StorageOps

# isort: split

import symforce.internal.symbolic as sf
from symforce import typing as T
from symforce import typing_util

from .impl.array_lie_group_ops import ArrayLieGroupOps
from .impl.databuffer_storage_ops import DataBufferStorageOps
from .impl.dataclass_lie_group_ops import DataclassLieGroupOps
from .impl.nonetype_lie_group_ops import NoneTypeLieGroupOps

# Register ops for scalars and sequences
from .impl.scalar_lie_group_ops import ScalarLieGroupOps
from .impl.sequence_lie_group_ops import SequenceLieGroupOps


class ScalarExpr(abc.ABC):
    """
    Metaclass for scalar expressions

    DataBuffer is a subclass of sf.Expr but we do not want it to be registered under
    ScalarLieGroupOps.
    """

    @abc.abstractmethod
    def __init__(self, *args: T.Any, **kwargs: T.Any) -> None:
        pass

    @classmethod
    def __subclasshook__(cls, subclass: T.Type) -> bool:
        if issubclass(subclass, sf.DataBuffer):
            return False
        return issubclass(subclass, sf.Expr) and isinstance(subclass, type)


for scalar_type in typing_util.SCALAR_TYPES:
    LieGroupOps.register(scalar_type, ScalarLieGroupOps)

LieGroupOps.register(ScalarExpr, ScalarLieGroupOps)

LieGroupOps.register(list, SequenceLieGroupOps)
LieGroupOps.register(tuple, SequenceLieGroupOps)

LieGroupOps.register(np.ndarray, ArrayLieGroupOps)

LieGroupOps.register(T.Dataclass, DataclassLieGroupOps)

# We register NoneType to allow dataclasses to have optional fields which default to "None".
LieGroupOps.register(type(None), NoneTypeLieGroupOps)

# TODO(hayk): Are these okay here or where can we put them? In theory we could just have this
# be automatic that if the given type has the methods that it gets registered automatically.
import sym

from .impl.sym_class_lie_group_ops import SymClassLieGroupOps

LieGroupOps.register(sym.Rot2, SymClassLieGroupOps)
LieGroupOps.register(sym.Rot3, SymClassLieGroupOps)
LieGroupOps.register(sym.Pose2, SymClassLieGroupOps)
LieGroupOps.register(sym.Pose3, SymClassLieGroupOps)

StorageOps.register(sf.DataBuffer, DataBufferStorageOps)
