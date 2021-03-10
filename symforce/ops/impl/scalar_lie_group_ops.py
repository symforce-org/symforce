import numpy as np

from symforce import python_util
from symforce import types as T
from symforce import sympy as sm

from .scalar_group_ops import ScalarGroupOps

if T.TYPE_CHECKING:
    from symforce import geo


class ScalarLieGroupOps(ScalarGroupOps):
    @staticmethod
    def tangent_dim(a: T.ScalarElementOrType) -> int:
        return 1

    @staticmethod
    def from_tangent(
        a: T.ScalarElementOrType, vec: T.List[T.Scalar], epsilon: T.Scalar
    ) -> T.ScalarElement:
        assert len(vec) == 1
        return sm.S(vec[0])

    @staticmethod
    def to_tangent(a: T.ScalarElement, epsilon: T.Scalar) -> T.List[T.Scalar]:
        return [a]

    @staticmethod
    def storage_D_tangent(a: T.ScalarElement) -> "geo.Matrix":
        from symforce import geo

        return geo.Matrix([1])

    @staticmethod
    def tangent_D_storage(a: T.ScalarElement) -> "geo.Matrix":
        from symforce import geo

        return geo.Matrix([1])

    @staticmethod
    def retract(a: T.ScalarElement, vec: T.List[T.Scalar], epsilon: T.Scalar) -> T.ScalarElement:
        return ScalarGroupOps.compose(a, ScalarLieGroupOps.from_tangent(a, vec, epsilon))

    @staticmethod
    def local_coordinates(
        a: T.ScalarElement, b: T.ScalarElement, epsilon: T.Scalar = 0
    ) -> T.List[T.Scalar]:
        return ScalarLieGroupOps.to_tangent(
            ScalarGroupOps.compose(ScalarGroupOps.inverse(a), b), epsilon
        )
