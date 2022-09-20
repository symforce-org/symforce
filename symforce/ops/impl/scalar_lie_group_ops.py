# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce.internal.symbolic as sf
from symforce import typing as T

from .abstract_vector_lie_group_ops import AbstractVectorLieGroupOps


class ScalarLieGroupOps(AbstractVectorLieGroupOps):
    @staticmethod
    def storage_dim(a: T.ScalarElementOrType) -> int:
        return 1

    @staticmethod
    def to_storage(a: T.ScalarElement) -> T.List[T.Scalar]:
        return [a]

    @staticmethod
    def from_storage(a: T.Any, elements: T.Sequence[T.Scalar]) -> T.ScalarElement:
        # NOTE: This returns a numeric type if both arguments are numeric types. If either argument
        # is a symbolic type, this returns a symbolic type.
        assert len(elements) == 1, "Scalar needs one element."
        if isinstance(a, type):
            if issubclass(a, sf.Expr) or isinstance(elements[0], sf.Expr):
                return sf.S(elements[0])
            return a(elements[0])  # type: ignore [call-arg]
        else:
            if isinstance(a, sf.Expr) or isinstance(elements[0], sf.Expr):
                return sf.S(elements[0])
            return type(a)(elements[0])

    @staticmethod
    def symbolic(a: T.Any, name: str, **kwargs: T.Dict) -> T.ScalarElement:
        return sf.Symbol(name, **kwargs)
