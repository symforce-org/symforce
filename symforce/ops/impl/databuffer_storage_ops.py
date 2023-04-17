# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce.internal.symbolic as sf
from symforce import typing as T


class DataBufferStorageOps:
    """
    Implements Storage operations for databuffers.

    For StorageOps, we choose a storage dim of 0 since the it is supposed to represent a pointer
    to external data, which we do not want to serialize
    """

    @staticmethod
    def storage_dim(a: sf.DataBuffer) -> int:
        return 0

    @staticmethod
    def to_storage(a: sf.DataBuffer) -> T.List[sf.DataBuffer]:
        return []

    @staticmethod
    def from_storage(a: sf.DataBuffer, elements: T.Sequence[sf.DataBuffer]) -> T.Element:
        raise NotImplementedError("Cannot restore DataBuffer from storage")

    @staticmethod
    def symbolic(a: sf.DataBuffer, name: str, **kwargs: T.Dict) -> sf.DataBuffer:
        return sf.DataBuffer(name, sf.Symbol(name + "_dim"))
