from symforce import types as T

Element = T.Any
ElementOrType = T.Union[Element, T.Type]


class ClassStorageOps(object):
    @staticmethod
    def storage_dim(a):
        # type: (ElementOrType) -> int
        return a.storage_dim()

    @staticmethod
    def to_storage(a):
        # type: (Element) -> T.List[T.Scalar]
        return a.to_storage()

    @staticmethod
    def from_storage(a, elements):
        # type: (ElementOrType, T.List[T.Scalar]) -> Element
        return a.from_storage(elements)

    @staticmethod
    def symbolic(a, name, **kwargs):
        # type: (ElementOrType, str, T.Dict) -> Element
        return a.symbolic(name, **kwargs)

    @staticmethod
    def evalf(a):
        # type: (Element) -> Element
        return a.evalf()
