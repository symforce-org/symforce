from __future__ import annotations

from symforce import typing as T

from .matrix_group_ops import MatrixGroupOps

if T.TYPE_CHECKING:
    from symforce.geo import Matrix

    # Specialization for matrix elements
    MatrixOrType = T.Union[Matrix, T.Type[Matrix]]


class MatrixLieGroupOps(MatrixGroupOps):
    """
    Implement lie group ops for geo.Matrix, treating it as the group R^{NxM} under addition
    """

    @staticmethod
    def tangent_dim(a: MatrixOrType) -> int:
        return a.storage_dim()

    @staticmethod
    def from_tangent(a: MatrixOrType, vec: T.Sequence[T.Scalar], epsilon: T.Scalar) -> Matrix:
        return a.from_storage(vec)

    @staticmethod
    def to_tangent(a: Matrix, epsilon: T.Scalar) -> T.List[T.Scalar]:
        return a.to_storage()

    @staticmethod
    def storage_D_tangent(a: Matrix) -> Matrix:
        return a.eye(a.storage_dim(), a.tangent_dim())

    @staticmethod
    def tangent_D_storage(a: Matrix) -> Matrix:
        return a.eye(a.tangent_dim(), a.storage_dim())

    @classmethod
    def retract(cls, a: Matrix, vec: T.Sequence[T.Scalar], epsilon: T.Scalar) -> Matrix:
        return a + cls.from_tangent(a, vec, epsilon)

    @classmethod
    def local_coordinates(cls, a: Matrix, b: Matrix, epsilon: T.Scalar) -> T.List[T.Scalar]:
        return cls.to_tangent(b - a, epsilon)
