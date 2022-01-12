from __future__ import annotations

from symforce import typing as T

from .class_group_ops import ClassGroupOps

if T.TYPE_CHECKING:
    from symforce.geo import Matrix

    # Specialization for matrix elements
    MatrixOrType = T.Union[Matrix, T.Type[Matrix]]


class MatrixGroupOps(ClassGroupOps):
    """
    Implement group ops for geo.Matrix, treating it as the group R^{NxM} under addition
    """

    @staticmethod
    def identity(a: MatrixOrType) -> Matrix:
        return a.zero()

    @staticmethod
    def compose(a: Matrix, b: Matrix) -> Matrix:
        return a + b

    @staticmethod
    def inverse(a: Matrix) -> Matrix:
        return -a
