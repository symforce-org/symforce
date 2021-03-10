from symforce import types as T

from .class_group_ops import ClassGroupOps

if T.TYPE_CHECKING:
    from symforce import geo


class ClassLieGroupOps(ClassGroupOps):
    @staticmethod
    def tangent_dim(a: T.ElementOrType) -> int:
        return a.tangent_dim()

    @staticmethod
    def from_tangent(a: T.ElementOrType, vec: T.List[T.Scalar], epsilon: T.Scalar) -> T.Element:
        return a.from_tangent(vec, epsilon)

    @staticmethod
    def to_tangent(a: T.Element, epsilon: T.Scalar) -> T.List[T.Scalar]:
        return a.to_tangent(epsilon)

    @staticmethod  # type: ignore
    def storage_D_tangent(a: T.Element) -> "geo.Matrix":
        return a.storage_D_tangent()

    @staticmethod  # type: ignore
    def tangent_D_storage(a: T.Element) -> "geo.Matrix":
        return a.tangent_D_storage()
