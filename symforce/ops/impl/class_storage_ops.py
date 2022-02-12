# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from symforce import typing as T


class ClassStorageOps:
    @staticmethod
    def storage_dim(a: T.ElementOrType) -> int:
        return a.storage_dim()

    @staticmethod
    def to_storage(a: T.Element) -> T.List[T.Scalar]:
        return a.to_storage()

    @staticmethod
    def from_storage(a: T.ElementOrType, elements: T.Sequence[T.Scalar]) -> T.Element:
        return a.from_storage(elements)

    @staticmethod
    def symbolic(a: T.ElementOrType, name: str, **kwargs: T.Dict) -> T.Element:
        return a.symbolic(name, **kwargs)
