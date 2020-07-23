from symforce.ops import GroupOps
from symforce.python_util import get_type
from symforce import types as T

from .sequence_storage_ops import SequenceStorageOps

Element = T.Sequence[T.Scalar]


class SequenceGroupOps(SequenceStorageOps):
    @staticmethod
    def identity(a):
        # type: (Element) -> Element
        return get_type(a)([GroupOps.identity(v) for v in a])

    @staticmethod
    def compose(a, b):
        # type: (Element, Element) -> Element
        return get_type(a)([GroupOps.compose(v_a, v_b) for v_a, v_b in zip(a, b)])

    @staticmethod
    def inverse(a):
        # type: (Element) -> Element
        return get_type(a)([GroupOps.inverse(v) for v in a])
