import numpy as np

from symforce.python_util import get_type
from symforce import sympy as sm
from symforce import types as T

from .ops import Ops

Element = T.Any
ElementOrType = T.Union[Element, T.Type]


class StorageOps(Ops):
    """
    API for symbolic data types that can be serialized to and from a vector of scalar quantities.
    """

    @staticmethod
    def storage_dim(a: ElementOrType) -> int:
        """
        Size of the element's storage, aka the number of scalar values it contains.
        """
        return Ops.implementation(get_type(a)).storage_dim(a)

    @staticmethod
    def to_storage(a: Element) -> T.List:
        """
        Serialization of the underlying storage into a list. This is NOT a tangent space.

        Args:
            a (Element):
        Returns:
            list: Length equal to `storage_dim(a)`
        """
        return Ops.implementation(get_type(a)).to_storage(a)

    @staticmethod
    def from_storage(a: ElementOrType, elements: T.Sequence[T.Scalar]) -> Element:
        """
        Construct from a flat list representation. Opposite of `.to_storage()`.
        """
        return Ops.implementation(get_type(a)).from_storage(a, elements)

    @staticmethod
    def symbolic(a: ElementOrType, name: str, **kwargs: T.Dict) -> Element:
        """
        Construct a symbolic element with the given name prefix.

        Args:
            a (Element or type):
            name (str): String prefix
            kwargs (dict): Additional arguments to pass to sm.Symbol (like assumptions)

        Returns:
            Storage:
        """
        return Ops.implementation(get_type(a)).symbolic(a, name, **kwargs)

    @staticmethod
    def evalf(a: Element) -> Element:
        """
        Evaluate to a numerical quantity (rationals, trig functions, etc).
        """
        return Ops.implementation(get_type(a)).evalf(a)

    @staticmethod
    def subs(a: Element, *args: T.Any, **kwargs: T.Any) -> Element:
        return StorageOps.from_storage(
            a, [sm.S(s).subs(*args, **kwargs) for s in StorageOps.to_storage(a)]
        )

    @staticmethod
    def simplify(a: Element) -> Element:
        return StorageOps.from_storage(a, sm.simplify(sm.Matrix(StorageOps.to_storage(a))))
