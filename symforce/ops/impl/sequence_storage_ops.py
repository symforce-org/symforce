# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from symforce import typing as T
from symforce.ops import StorageOps
from symforce.typing_util import get_type


class SequenceStorageOps:
    @staticmethod
    def storage_dim(a: T.SequenceElement) -> int:
        return sum(StorageOps.storage_dim(v) for v in a)

    @staticmethod
    def to_storage(a: T.SequenceElement) -> T.List[T.Scalar]:
        return [scalar for v in a for scalar in StorageOps.to_storage(v)]

    @staticmethod
    def from_storage(a: T.SequenceElement, elements: T.Sequence[T.Scalar]) -> T.SequenceElement:
        assert len(elements) == SequenceStorageOps.storage_dim(a)
        new_a = []
        inx = 0
        for v in a:
            dim = StorageOps.storage_dim(v)
            new_a.append(StorageOps.from_storage(v, elements[inx : inx + dim]))
            inx += dim

        return get_type(a)(new_a)

    @staticmethod
    def symbolic(a: T.SequenceElement, name: str, **kwargs: T.Dict) -> T.SequenceElement:
        return get_type(a)(
            [StorageOps.symbolic(v, f"{name}_{i}", **kwargs) for i, v in enumerate(a)]
        )
