# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import numpy as np

from symforce import typing as T
from symforce.ops import StorageOps


class ArrayStorageOps:
    """
    Implements Storage operations for numpy ndarrays.
    """

    @staticmethod
    def storage_dim(a: T.ArrayElementOrType) -> int:
        # NOTE(brad): Must take T.ArrayElementOrType to match AbstractStorageOps
        assert isinstance(a, np.ndarray)
        return a.size

    @staticmethod
    def to_storage(a: T.ArrayElement) -> T.List[T.Scalar]:
        # NOTE(brad): I have the T.cast because mypy thinks the values of np.nditer are tuples.
        return [
            T.cast(np.ndarray, scalar)[()] for scalar in np.nditer(a, order="F", flags=["refs_ok"])
        ]

    @staticmethod
    def from_storage(a: T.ArrayElementOrType, elements: T.Sequence[T.Scalar]) -> T.ArrayElement:
        # NOTE(brad): Must take T.ArrayElementOrType to match AbstractStorageOps
        assert isinstance(a, np.ndarray)
        assert len(elements) == ArrayStorageOps.storage_dim(a)

        return np.array(elements).reshape(tuple(reversed(a.shape))).transpose()

    @staticmethod
    def symbolic(a: T.ArrayElementOrType, name: str, **kwargs: T.Dict) -> T.ArrayElement:
        # NOTE(brad): Must take T.ArrayElementOrType to match AbstractStorageOps
        assert isinstance(a, np.ndarray)
        return np.array(
            [StorageOps.symbolic(v, f"{name}_{i}", **kwargs) for i, v in enumerate(a)]
        ).reshape(a.shape)
