# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce.internal.symbolic as sf
from symforce import typing as T
from symforce.typing_util import get_type
from symforce.typing_util import scalar_like

from .ops import Ops


class StorageOps(Ops):
    """
    API for symbolic data types that can be serialized to and from a vector of scalar quantities.
    """

    @staticmethod
    def storage_dim(a: T.ElementOrType) -> int:
        """
        Size of the element's storage, aka the number of scalar values it contains.
        """
        return StorageOps.implementation(get_type(a)).storage_dim(a)

    @staticmethod
    def to_storage(a: T.Element) -> T.List:
        """
        Serialization of the underlying storage into a list. This is NOT a tangent space.

        Args:
            a:
        Returns:
            list: Length equal to `storage_dim(a)`
        """
        return StorageOps.implementation(get_type(a)).to_storage(a)

    @staticmethod
    def from_storage(a: T.ElementOrType, elements: T.Sequence[T.Scalar]) -> T.Element:
        """
        Construct from a flat list representation. Opposite of `.to_storage()`.
        """
        return StorageOps.implementation(get_type(a)).from_storage(a, elements)

    @staticmethod
    def symbolic(a: T.ElementOrType, name: str, **kwargs: T.Dict) -> T.Element:
        """
        Construct a symbolic element with the given name prefix.

        Args:
            a:
            name: String prefix
            kwargs: Additional arguments to pass to sf.Symbol (like assumptions)

        Returns:
            Storage:
        """
        return StorageOps.implementation(get_type(a)).symbolic(a, name, **kwargs)

    @staticmethod
    def evalf(a: T.Element) -> T.Element:
        """
        Evaluate to a numerical quantity (rationals, trig functions, etc).
        """

        def evalf_scalar(s: T.Scalar) -> T.Scalar:
            if hasattr(s, "evalf"):
                return s.evalf()  # type: ignore
            if scalar_like(s):
                return s
            raise TypeError

        return StorageOps.from_storage(a, [evalf_scalar(s) for s in StorageOps.to_storage(a)])

    @staticmethod
    def subs(a: T.Element, *args: T.Any, **kwargs: T.Any) -> T.Element:
        # We convert to a Matrix here so that we can call `.subs` once, which is faster
        return StorageOps.from_storage(
            a, list(iter(sf.sympy.Matrix(StorageOps.to_storage(a)).subs(*args, **kwargs)))
        )

    @staticmethod
    def simplify(a: T.Element) -> T.Element:
        return StorageOps.from_storage(
            a, list(sf.simplify(sf.sympy.Matrix(StorageOps.to_storage(a))))
        )
