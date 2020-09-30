import numpy as np

from symforce import python_util
from symforce import types as T
from symforce import sympy as sm

from .scalar_group_ops import ScalarGroupOps

Element = T.Scalar
ElementOrType = T.Union[Element, T.Type]


class ScalarLieGroupOps(ScalarGroupOps):
    @staticmethod
    def tangent_dim(a):
        # type: (ElementOrType) -> int
        return 1

    @staticmethod
    def from_tangent(a, vec, epsilon):
        # type: (ElementOrType, T.List[T.Scalar], T.Scalar) -> Element
        assert len(vec) == 1
        return sm.S(vec[0])

    @staticmethod
    def to_tangent(a, epsilon):
        # type: (Element, T.Scalar) -> T.List[T.Scalar]
        return [a]

    @staticmethod  # type: ignore
    def storage_D_tangent(a):
        # type: (Element) -> geo.Matrix
        from symforce import geo

        return geo.Matrix([1])

    @staticmethod  # type: ignore
    def tangent_D_storage(a):
        # type: (Element) -> geo.Matrix
        from symforce import geo

        return geo.Matrix([1])
