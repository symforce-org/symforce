import numpy as np

from symforce.ops import StorageOps
from symforce import python_util
from symforce import types as T
from symforce import sympy as sm

Element = T.Sequence[T.Scalar]


class SequenceStorageOps(object):
    @staticmethod
    def storage_dim(a):
        # type: (Element) -> int
        return len(a)

    @staticmethod
    def to_storage(a):
        # type: (Element) -> T.List[T.Scalar]
        return list(a)

    @staticmethod
    def from_storage(a, elements):
        # type: (Element, T.List[T.Scalar]) -> Element
        assert len(elements) == SequenceStorageOps.storage_dim(a)
        return elements

    @staticmethod
    def symbolic(a, name, **kwargs):
        # type: (Element, str, T.Dict) -> Element
        return [
            sm.Symbol("{}_{}".format(name, i), **kwargs)
            for i in range(SequenceStorageOps.storage_dim(a))
        ]

    @staticmethod
    def evalf(a):
        # type: (Element) -> Element
        return [StorageOps.evalf(v) for v in a]
