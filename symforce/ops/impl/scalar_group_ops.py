from symforce import types as T

from .scalar_storage_ops import ScalarStorageOps

Element = T.Scalar
ElementOrType = T.Union[Element, T.Type]


class ScalarGroupOps(ScalarStorageOps):
    @staticmethod
    def identity(a: ElementOrType) -> Element:
        return 0.0

    @staticmethod
    def compose(a: Element, b: Element) -> Element:
        return a + b

    @staticmethod
    def inverse(a: Element) -> Element:
        return -a
