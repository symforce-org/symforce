# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
from symforce import typing as T
from symforce.ops.impl.abstract_vector_lie_group_ops import AbstractVectorLieGroupOps


class NoneTypeLieGroupOps(AbstractVectorLieGroupOps):
    """
    Class for implementing ops on "None" object. This is primarily used when performing ops
    on dataclasses with optional fields which could be "None".

    AbstractVectorLieGroupOps lets us only implement the StorageOps functions, and use those
    functions to perform Group and LieGroup ops.
    """

    @staticmethod
    def storage_dim(a: T.Any) -> int:
        return 0

    @staticmethod
    def to_storage(a: T.Any) -> T.List[T.Scalar]:
        return []

    @staticmethod
    def from_storage(a: T.Any, elements: T.Sequence[T.Scalar]) -> None:
        assert len(elements) == 0
        return None

    @staticmethod
    def symbolic(a: T.Any, name: str, **kwargs: T.Dict) -> None:
        return None
