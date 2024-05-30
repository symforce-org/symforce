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

    :class:`symforce.symbolic.DataBuffer` is a subclass of :class:`sf.Expr <symforce.symbolic.Expr>`
    but we do not want it to be registered under
    :class:`ScalarLieGroupOps <symforce.ops.impl.scalar_lie_group_ops.ScalarLieGroupOps>`.
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


class LieGroupSymClass(abc.ABC):
    """
    Metaclass for generated numeric geo classes

    We use a metaclass here to avoid having `symforce.ops` depend on `sym`, which would make it
    impossible to generate `sym` from scratch.

    TODO(aaron): SymClassLieGroupOps does the wrong thing for some methods, e.g. storage_D_tangent
    returns the wrong type.  We should also implement this for the cam classes, which aren't lie
    groups.
    """

    @staticmethod
    def __subclasshook__(subclass: T.Type) -> bool:
        for parent in subclass.__mro__:
            if parent.__module__.startswith("sym."):
                from symforce import geo
                from symforce.ops.interfaces.lie_group import LieGroup

                maybe_geo_class = getattr(geo, parent.__name__, None)
                if maybe_geo_class is not None and issubclass(maybe_geo_class, LieGroup):
                    return True
        return False


from .impl.sym_class_lie_group_ops import SymClassLieGroupOps

LieGroupOps.register(LieGroupSymClass, SymClassLieGroupOps)

StorageOps.register(sf.DataBuffer, DataBufferStorageOps)
