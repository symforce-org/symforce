from symforce import typing as T
from symforce import sympy as sm

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
        assert len(elements) == 1, "Scalar needs one element."
        return sm.S(elements[0])

    @staticmethod
    def symbolic(a: T.Any, name: str, **kwargs: T.Dict) -> T.ScalarElement:
        return sm.Symbol(name, **kwargs)
