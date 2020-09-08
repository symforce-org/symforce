from symforce import python_util
from symforce import types as T
from symforce import sympy as sm

Element = T.Scalar
ElementOrType = T.Union[Element, T.Type]


class ScalarStorageOps(object):
    @staticmethod
    def storage_dim(a):
        # type: (T.Any) -> int
        return 1

    @staticmethod
    def to_storage(a):
        # type: (Element) -> T.List[T.Scalar]
        return [a]

    @staticmethod
    def from_storage(a, elements):
        # type: (T.Any, T.List[T.Scalar]) -> Element
        assert len(elements) == 1, "Scalar needs one element."
        return sm.S(elements[0])

    @staticmethod
    def symbolic(a, name, **kwargs):
        # type: (T.Any, str, T.Dict) -> Element
        return sm.Symbol(name, **kwargs)

    @staticmethod
    def evalf(a):
        # type: (Element) -> Element
        if hasattr(a, "evalf"):
            return a.evalf()  # type: ignore
        if python_util.scalar_like(a):
            return a
        raise TypeError
