import numpy as np

from symforce.ops import StorageOps
from symforce.python_util import get_type
from symforce import typing as T
from symforce import sympy as sm


class SequenceStorageOps:
    @staticmethod
    def storage_dim(a: T.SequenceElement) -> int:
        return sum([StorageOps.storage_dim(v) for v in a])

    @T.overload
    @staticmethod
    def to_storage(a: T.Sequence[T.Element]) -> T.List[T.Scalar]:  # pragma: no cover
        """
        Overload so that mypy knows a Sequence[Element] argument results in a List[Scalar]
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
    def to_storage(a: T.SequenceElement) -> T.Union[T.List[T.Scalar], np.ndarray]:
        flat_list = [scalar for v in a for scalar in StorageOps.to_storage(v)]
        if isinstance(a, np.ndarray):
            return np.asarray(flat_list)
        return get_type(a)(flat_list)

    @staticmethod
    def from_storage(a: T.SequenceElement, elements: T.List[T.Scalar]) -> T.SequenceElement:
        assert len(elements) == SequenceStorageOps.storage_dim(a)
        new_a = []
        inx = 0
        for v in a:
            dim = StorageOps.storage_dim(v)
            new_a.append(StorageOps.from_storage(v, elements[inx : inx + dim]))
            inx += dim

        a_type = get_type(a)
        if a_type == np.ndarray:
            assert isinstance(a, np.ndarray)
            return np.array(new_a).reshape(a.shape)
        else:
            return a_type(new_a)

    @staticmethod
    def symbolic(a: T.SequenceElement, name: str, **kwargs: T.Dict) -> T.SequenceElement:
        return get_type(a)(
            [StorageOps.symbolic(v, f"{name}_{i}", **kwargs) for i, v in enumerate(a)]
        )
