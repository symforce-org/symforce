# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import abc

from symforce import typing as T


class AbstractStorageOps(abc.ABC):
    """
    An abstract base class for StorageOps implementations.

    Useful for when multiple classes can implement their GroupOps and LieGroupOps
    implementations in terms of their storage ops in the same manner, despite having
    different StorageOps impelmentations.

    For example, classes whose storage spaces are abstract vector spaces and whose
    group operations are their vector operations. See abstract_vector_group_ops.py.
    """

    @staticmethod
    @abc.abstractmethod
    def storage_dim(a: T.ElementOrType) -> int:
        pass

    @staticmethod
    @abc.abstractmethod
    def to_storage(a: T.Element) -> T.List[T.Scalar]:
        pass

    @staticmethod
    @abc.abstractmethod
    def from_storage(a: T.ElementOrType, elements: T.Sequence[T.Scalar]) -> T.Element:
        pass

    @staticmethod
    @abc.abstractmethod
    def symbolic(a: T.ElementOrType, name: str, **kwargs: T.Dict) -> T.Element:
        pass
