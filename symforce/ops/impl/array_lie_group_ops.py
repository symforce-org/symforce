# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from symforce import typing as T

from .abstract_vector_lie_group_ops import AbstractVectorLieGroupOps
from .array_storage_ops import ArrayStorageOps


class ArrayLieGroupOps(ArrayStorageOps, AbstractVectorLieGroupOps[T.ArrayElement]):
    """
    Implements Lie Group operations for numpy ndarrays.
    """
