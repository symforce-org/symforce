# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from symforce import ops
from symforce import typing as T

from .storage import Storage


class Group(Storage):
    """
    Interface for objects that implement the group concept. Because this
    class is registered using ClassGroupOps (see bottom of this file), any
    object that inherits from Group and that implements the functions defined
    in this class can be used with the GroupOps concept.

    Because Group is a subclass of Storage, objects inheriting from Group
    must also implement the functions defined in Storage (e.g. storage_dim,
    to_storage, etc.), and can also be used with StorageOps.
    """

    # Type that represents this or any subclasses
    GroupT = T.TypeVar("GroupT", bound="Group")

    @classmethod
    def identity(cls: T.Type[GroupT]) -> GroupT:
        """
        Identity element such that `compose(a, identity) = a`.
        """
        raise NotImplementedError()

    def compose(self: GroupT, other: GroupT) -> GroupT:
        """
        Apply the group operation with other.
        """
        raise NotImplementedError()

    def inverse(self: GroupT) -> GroupT:
        """
        Group inverse, such that `compose(a, inverse(a)) = identity`.
        """
        raise NotImplementedError()

    def between(self: GroupT, b: GroupT) -> GroupT:
        """
        Returns the element that when composed with self produces b. For vector spaces it is `b - self`.

        Implementation is simply `compose(inverse(self), b)`.
        """
        return self.inverse().compose(b)


from ..impl.class_group_ops import ClassGroupOps

ops.GroupOps.register(Group, ClassGroupOps)
