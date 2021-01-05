from symforce import types as T
from symforce.python_util import get_type

from .ops import Ops
from .storage_ops import StorageOps

Element = T.Any
ElementOrType = T.Union[Element, T.Type]


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
    def identity(a: ElementOrType) -> Element:
        """
        Identity element of the given type's group.

        This method does not rely on the value of a, only the type.

        Args:
            a (ElementOrType):

        Returns:
            Element: b such that a @ b = a
        """
        return Ops.implementation(get_type(a)).identity(a)

    @staticmethod
    def compose(a: Element, b: Element) -> Element:
        """
        Composition of two elements in the group.

        Args:
            a (Element):
            b (Element):

        Returns:
            Element: a @ b
        """
        return Ops.implementation(get_type(a)).compose(a, b)

    @staticmethod
    def inverse(a: Element) -> Element:
        """
        Inverse of the element a.

        Args:
            a (Element):

        Returns:
            Element: b such that a @ b = identity
        """
        return Ops.implementation(get_type(a)).inverse(a)

    @staticmethod
    def between(a: Element, b: Element) -> Element:
        """
        Returns the element that when composed with a produces b. For vector spaces it is b - a.

        Implementation is simply `compose(inverse(a), b)`.

        Args:
            a (Element):
            b (Element):

        Returns:
            Element: c such that a @ c = b
        """
        return GroupOps.compose(GroupOps.inverse(a), b)
