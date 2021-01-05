from symforce import python_util
from symforce import types as T
from symforce import sympy as sm

Element = T.Scalar
ElementOrType = T.Union[Element, T.Type]


class ScalarStorageOps:
    @staticmethod
    def storage_dim(a: T.Any) -> int:
        return 1

    @staticmethod
    def to_storage(a: Element) -> T.List[T.Scalar]:
        return [a]

    @staticmethod
    def from_storage(a: T.Any, elements: T.List[T.Scalar]) -> Element:
        assert len(elements) == 1, "Scalar needs one element."
        return sm.S(elements[0])

    @staticmethod
    def symbolic(a: T.Any, name: str, **kwargs: T.Dict) -> Element:
        return sm.Symbol(name, **kwargs)

    @staticmethod
    def evalf(a: Element) -> Element:
        if hasattr(a, "evalf"):
            return a.evalf()  # type: ignore
        if python_util.scalar_like(a):
            return a
        raise TypeError
