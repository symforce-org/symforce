import numpy as np

from symforce.ops import StorageOps
from symforce.python_util import get_type
from symforce import types as T
from symforce import sympy as sm

Element = T.Sequence[T.Scalar]


class SequenceStorageOps:
    @staticmethod
    def storage_dim(a: Element) -> int:
        return sum([StorageOps.storage_dim(v) for v in a])

    @T.overload
    @staticmethod
    def to_storage(a: Element) -> T.List[T.Scalar]:  # pragma: no cover
        """
        Overload so that mypy knows an Element argument results in a List[Scalar]
        """
        pass

    @T.overload
    @staticmethod
    def to_storage(a: np.ndarray) -> np.ndarray:  # pragma: no cover
        """
        Overload so that mypy knows an ndarray argument results in an ndarray
        """
        pass

    @staticmethod
    def to_storage(a: T.Union[Element, np.ndarray]) -> T.Union[T.List[T.Scalar], np.ndarray]:
        flat_list = [scalar for v in a for scalar in StorageOps.to_storage(v)]
        if isinstance(a, np.ndarray):
            return np.asarray(flat_list)
        return get_type(a)(flat_list)

    @staticmethod
    def from_storage(a: Element, elements: T.List[T.Scalar]) -> Element:
        assert len(elements) == SequenceStorageOps.storage_dim(a)
        new_a = get_type(a)()
        inx = 0
        for v in a:
            dim = StorageOps.storage_dim(v)
            new_a.append(StorageOps.from_storage(v, elements[inx : inx + dim]))
            inx += dim
        return new_a

    @staticmethod
    def symbolic(a: Element, name: str, **kwargs: T.Dict) -> Element:
        return get_type(a)(
            [StorageOps.symbolic(v, f"{name}_{i}", **kwargs) for i, v in enumerate(a)]
        )

    @T.overload
    @staticmethod
    def evalf(a: Element) -> Element:  # pragma: no cover
        """
        Overload so that mypy knows an Element argument results in a List[Scalar]
        """
        pass

    @T.overload
    @staticmethod
    def evalf(a: np.ndarray) -> np.ndarray:  # pragma: no cover
        """
        Overload so that mypy knows an ndarray argument results in an ndarray
        """
        pass

    @staticmethod
    def evalf(a: T.Union[Element, np.ndarray]) -> T.Union[Element, np.ndarray]:
        flat_list = [StorageOps.evalf(v) for v in a]
        if isinstance(a, np.ndarray):
            return np.asarray(flat_list)
        return get_type(a)(flat_list)
