# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import numpy as np

from symforce.ops import StorageOps
from symforce import typing as T


class ArrayStorageOps:
    """
    Implements Storage operations for numpy ndarrays.
    """

    @staticmethod
    def storage_dim(a: T.ArrayElementOrType) -> int:
        # NOTE(brad): Must take T.ArrayElementOrType to match AbstractStorageOps
        assert isinstance(a, np.ndarray)
        return sum([StorageOps.storage_dim(v) for v in a])

    @staticmethod
    def to_storage(a: T.ArrayElement) -> T.List[T.Scalar]:
        return [scalar for v in a for scalar in StorageOps.to_storage(v)]

    @staticmethod
    def from_storage(a: T.ArrayElementOrType, elements: T.Sequence[T.Scalar]) -> T.ArrayElement:
        # NOTE(brad): Must take T.ArrayElementOrType to match AbstractStorageOps
        assert isinstance(a, np.ndarray)
        assert len(elements) == ArrayStorageOps.storage_dim(a)
        new_a = []
        inx = 0
        for v in a:
            dim = StorageOps.storage_dim(v)
            new_a.append(StorageOps.from_storage(v, elements[inx : inx + dim]))
            inx += dim

        return np.array(new_a).reshape(a.shape)

    @staticmethod
    def symbolic(a: T.ArrayElementOrType, name: str, **kwargs: T.Dict) -> T.ArrayElement:
        # NOTE(brad): Must take T.ArrayElementOrType to match AbstractStorageOps
        assert isinstance(a, np.ndarray)
        return np.array(
            [StorageOps.symbolic(v, f"{name}_{i}", **kwargs) for i, v in enumerate(a)]
        ).reshape(a.shape)
