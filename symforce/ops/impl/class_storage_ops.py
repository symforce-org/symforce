from symforce import types as T

Element = T.Any
ElementOrType = T.Union[Element, T.Type]


class ClassStorageOps:
    @staticmethod
    def storage_dim(a: ElementOrType) -> int:
        return a.storage_dim()

    @staticmethod
    def to_storage(a: Element) -> T.List[T.Scalar]:
        return a.to_storage()

    @staticmethod
    def from_storage(a: ElementOrType, elements: T.List[T.Scalar]) -> Element:
        return a.from_storage(elements)

    @staticmethod
    def symbolic(a: ElementOrType, name: str, **kwargs: T.Dict) -> Element:
        return a.symbolic(name, **kwargs)

    @staticmethod
    def evalf(a: Element) -> Element:
        return a.evalf()
