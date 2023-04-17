# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from symforce import typing as T
from symforce.typing_util import get_type

from .storage_ops import StorageOps


class GroupOps(StorageOps):
    """
    API for mathematical groups.

    A group is an algebraic structure consisting of a set of elements equipped with an operation
    that combines any two elements to form a third element and that satisfies four conditions
    called the group axioms - closure, associativity, identity and invertibility.

    Reference:

        https://en.wikipedia.org/wiki/Group_(mathematics)
    """

    @staticmethod
    def identity(a: T.ElementOrType) -> T.Element:
        """
        Identity element of the given type's group.

        This method does not rely on the value of a, only the type.

        Returns:
            Element: b such that a @ b = a
        """
        return GroupOps.implementation(get_type(a)).identity(a)

    @staticmethod
    def compose(a: T.Element, b: T.Element) -> T.Element:
        """
        Composition of two elements in the group.

        Returns:
            Element: a @ b
        """
        return GroupOps.implementation(get_type(a)).compose(a, b)

    @staticmethod
    def inverse(a: T.Element) -> T.Element:
        """
        Inverse of the element a.

        Returns:
            Element: b such that a @ b = identity
        """
        return GroupOps.implementation(get_type(a)).inverse(a)

    @staticmethod
    def between(a: T.Element, b: T.Element) -> T.Element:
        """
        Returns the element that when composed with a produces b. For vector spaces it is b - a.

        Implementation is simply `compose(inverse(a), b)`.

        Returns:
            Element: c such that a @ c = b
        """
        return GroupOps.compose(GroupOps.inverse(a), b)
