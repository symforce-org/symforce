# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from symforce import sympy as sm

from symforce import typing as T


class DataBufferStorageOps:
    """
    Implements Storage operations for databuffers.

    For StorageOps, we choose a storage dim of 0 since the it is supposed to represent a pointer
    to external data, which we do not want to serialize
    """

    @staticmethod
    def storage_dim(a: sm.DataBuffer) -> int:
        return 0

    @staticmethod
    def to_storage(a: sm.DataBuffer) -> T.List[sm.DataBuffer]:
        return []

    @staticmethod
    def from_storage(a: sm.DataBuffer, elements: T.Sequence[sm.DataBuffer]) -> T.Element:
        raise NotImplementedError("Cannot restore DataBuffer from storage")

    @staticmethod
    def symbolic(a: sm.DataBuffer, name: str, **kwargs: T.Dict) -> sm.DataBuffer:
        return sm.DataBuffer(name, sm.S(name + "_dim"))
