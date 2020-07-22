from symforce import types as T

from .class_group_ops import ClassGroupOps

Element = T.Any
ElementOrType = T.Union[Element, T.Type]


class ClassLieGroupOps(ClassGroupOps):
    @staticmethod
    def tangent_dim(a):
        # type: (ElementOrType) -> int
        return a.tangent_dim()

    @staticmethod
    def from_tangent(a, vec, epsilon):
        # type: (ElementOrType, T.List[T.Scalar], T.Scalar) -> Element
        return a.from_tangent(vec, epsilon)

    @staticmethod
    def to_tangent(a, epsilon):
        # type: (Element, T.Scalar) -> T.List[T.Scalar]
        return a.to_tangent(epsilon)

    @staticmethod  # type: ignore
    def storage_D_tangent(a):
        # type: (Element) -> geo.Matrix
        return a.storage_D_tangent()
