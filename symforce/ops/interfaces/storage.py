# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce.internal.symbolic as sf
from symforce import ops
from symforce import typing as T


class Storage:
    """
    Interface for objects that implement the storage concept. Because this
    class is registered using ClassStorageOps (see bottom of this file), any
    object that inherits from Storage and that implements the functions defined
    in this class can be used with the StorageOps concept.

    I.e. calling
        ops.StorageOps.storage_dim(my_obj)

    will return the same result as my_obj.storage_dim() if my_obj inherits from
    this class.
    """

    # Type that represents this or any subclasses
    StorageT = T.TypeVar("StorageT", bound="Storage")

    @classmethod
    def storage_dim(cls) -> int:
        """
        Dimension of underlying storage
        """
        raise NotImplementedError()

    def __repr__(self: StorageT) -> str:
        """
        String representation of this type.
        """
        raise NotImplementedError()

    def to_storage(self: StorageT) -> T.List[T.Scalar]:
        """
        Flat list representation of the underlying storage, length of STORAGE_DIM.
        This is used purely for plumbing, it is NOT like a tangent space.
        """
        raise NotImplementedError()

    @classmethod
    def from_storage(cls: T.Type[StorageT], elements: T.Sequence[T.Scalar]) -> StorageT:
        """
        Construct from a flat list representation. Opposite of `.to_storage()`.
        """
        raise NotImplementedError()

    def __eq__(self: StorageT, other: T.Any) -> bool:
        """
        Returns exact equality between self and other.
        """
        if not isinstance(self, other.__class__):
            return False

        self_list, other_list = self.to_storage(), other.to_storage()
        if not len(self_list) == len(other_list):
            return False

        if not all(s == o for s, o in zip(self_list, other_list)):
            return False

        return True

    def subs(self: StorageT, *args: T.Any, **kwargs: T.Any) -> StorageT:
        """
        Substitute given values of each scalar element into a new instance.
        """
        return ops.StorageOps.subs(self, *args, **kwargs)

    # TODO(hayk): Way to get sf.simplify to work on these types directly?
    def simplify(self: StorageT) -> StorageT:
        """
        Simplify each scalar element into a new instance.
        """
        return self.from_storage(sf.simplify(sf.sympy.Matrix(self.to_storage())))

    @classmethod
    def symbolic(cls: T.Type[StorageT], name: str, **kwargs: T.Any) -> StorageT:
        """
        Construct a symbolic element with the given name prefix. Kwargs are forwarded
        to sf.Symbol (for example, sympy assumptions).
        """
        return cls.from_storage(
            [sf.Symbol(f"{name}_{i}", **kwargs) for i in range(cls.storage_dim())]
        )

    def evalf(self: StorageT) -> StorageT:
        """
        Numerical evaluation.
        """
        return self.from_storage([ops.StorageOps.evalf(e) for e in self.to_storage()])

    def __hash__(self) -> int:
        """
        Hash this object in immutable form, by combining all their scalar hashes.

        NOTE(hayk, nathan): This is somewhat dangerous because we don't always guarantee
        that Storage objects are immutable (e.g. sf.Matrix). If you add this object as
        a key to a dict, modify it, and access the dict, it will show up as another key
        because it breaks the abstraction that an object will maintain the same hash over
        its lifetime.
        """
        return tuple(self.to_storage()).__hash__()


from ..impl.class_storage_ops import ClassStorageOps

ops.StorageOps.register(Storage, ClassStorageOps)
