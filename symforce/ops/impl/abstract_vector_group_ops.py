from .abstract_storage_ops import AbstractStorageOps

from symforce import typing as T


class AbstractVectorGroupOps(AbstractStorageOps):
    """
    An abstract base class for GroupOps implementations whose group operation
    is equivalent to storage representation addition, and whose identity element
    is the element whose storage representation is the 0 vector.

    For a list of abstract methods which child classes must define, see
    AbstractStorageOps from abstract_storage_ops.py
    """

    @classmethod
    def identity(cls, a: T.ElementOrType) -> T.Element:
        return cls.from_storage(a, [0] * cls.storage_dim(a))

    @classmethod
    def compose(cls, a: T.Element, b: T.Element) -> T.Element:
        if cls.storage_dim(a) != cls.storage_dim(b):
            raise ValueError(
                f"Elements must have the same storage length ({cls.storage_dim(a)} != {cls.storage_dim(b)})."
            )
        return cls.from_storage(
            a, [ax + bx for ax, bx in zip(cls.to_storage(a), cls.to_storage(b))]
        )

    @classmethod
    def inverse(cls, a: T.Element) -> T.Element:
        return cls.from_storage(a, [-x for x in cls.to_storage(a)])
