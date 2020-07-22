from symforce import types as T

from .scalar_storage_ops import ScalarStorageOps

Element = T.Scalar
ElementOrType = T.Union[Element, T.Type]


class ScalarGroupOps(ScalarStorageOps):
    @staticmethod
    def identity(a):
        # type: (ElementOrType) -> Element
        return 0.0

    @staticmethod
    def compose(a, b):
        # type: (Element, Element) -> Element
        return a + b

    @staticmethod
    def inverse(a):
        # type: (Element) -> Element
        return -a
