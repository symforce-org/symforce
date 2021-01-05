import numpy as np

from symforce import python_util
from symforce import types as T
from symforce import sympy as sm

from .scalar_group_ops import ScalarGroupOps

Element = T.Scalar
ElementOrType = T.Union[Element, T.Type]

if T.TYPE_CHECKING:
    from symforce import geo


class ScalarLieGroupOps(ScalarGroupOps):
    @staticmethod
    def tangent_dim(a: ElementOrType) -> int:
        return 1

    @staticmethod
    def from_tangent(a: ElementOrType, vec: T.List[T.Scalar], epsilon: T.Scalar) -> Element:
        assert len(vec) == 1
        return sm.S(vec[0])

    @staticmethod
    def to_tangent(a: Element, epsilon: T.Scalar) -> T.List[T.Scalar]:
        return [a]

    @staticmethod  # type: ignore
    def storage_D_tangent(a: Element) -> "geo.Matrix":
        from symforce import geo

        return geo.Matrix([1])

    @staticmethod  # type: ignore
    def tangent_D_storage(a: Element) -> "geo.Matrix":
        from symforce import geo

        return geo.Matrix([1])
