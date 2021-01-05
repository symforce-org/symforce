from symforce import types as T

from .class_storage_ops import ClassStorageOps

Element = T.Any
ElementOrType = T.Union[Element, T.Type]


class ClassGroupOps(ClassStorageOps):
    @staticmethod
    def identity(a: ElementOrType) -> Element:
        return a.identity()

    @staticmethod
    def compose(a: Element, b: Element) -> Element:
        return a.compose(b)

    @staticmethod
    def inverse(a: Element) -> Element:
        return a.inverse()
