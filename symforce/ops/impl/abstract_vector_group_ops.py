# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from symforce import typing as T

from .abstract_storage_ops import AbstractStorageOps

ElementT = T.TypeVar("ElementT")
ElementOrTypeT = T.Union[ElementT, T.Type[ElementT]]


class AbstractVectorGroupOps(AbstractStorageOps[ElementT]):
    """
    An abstract base class for GroupOps implementations whose group operation
    is equivalent to storage representation addition, and whose identity element
    is the element whose storage representation is the 0 vector.

    For a list of abstract methods which child classes must define, see
    AbstractStorageOps from abstract_storage_ops.py
    """

    @classmethod
    def identity(cls, a: ElementOrTypeT) -> ElementT:
        return cls.from_storage(a, [0] * cls.storage_dim(a))

    @classmethod
    def compose(cls, a: ElementT, b: ElementT) -> ElementT:
        if cls.storage_dim(a) != cls.storage_dim(b):
            raise ValueError(
                f"Elements must have the same storage length ({cls.storage_dim(a)} != {cls.storage_dim(b)})."
            )
        return cls.from_storage(
            a, [ax + bx for ax, bx in zip(cls.to_storage(a), cls.to_storage(b))]
        )

    @classmethod
    def inverse(cls, a: ElementT) -> ElementT:
        return cls.from_storage(a, [-x for x in cls.to_storage(a)])
