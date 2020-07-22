from symforce import types as T

from .class_storage_ops import ClassStorageOps

Element = T.Any
ElementOrType = T.Union[Element, T.Type]


class ClassGroupOps(ClassStorageOps):
    @staticmethod
    def identity(a):
        # type: (ElementOrType) -> Element
        return a.identity()

    @staticmethod
    def compose(a, b):
        # type: (Element, Element) -> Element
        return a.compose(b)

    @staticmethod
    def inverse(a):
        # type: (Element) -> Element
        return a.inverse()
