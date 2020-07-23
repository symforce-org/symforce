from symforce.ops import StorageOps
from symforce.ops import LieGroupOps
from symforce.python_util import get_type
from symforce import types as T

from .sequence_group_ops import SequenceGroupOps

Element = T.Sequence[T.Scalar]


class SequenceLieGroupOps(SequenceGroupOps):
    @staticmethod
    def tangent_dim(a):
        # type: (Element) -> int
        return sum([LieGroupOps.tangent_dim(v) for v in a])

    @staticmethod
    def from_tangent(a, vec, epsilon):
        # type: (Element, T.List[T.Scalar], T.Scalar) -> Element
        assert len(vec) == SequenceLieGroupOps.tangent_dim(a)
        new_a = get_type(a)()
        inx = 0
        for v in a:
            dim = LieGroupOps.tangent_dim(v)
            new_a.append(LieGroupOps.from_tangent(v, vec[inx : inx + dim], epsilon))
            inx += dim
        return new_a

    @staticmethod
    def to_tangent(a, epsilon):
        # type: (Element, T.Scalar) -> T.List[Element]
        return get_type(a)([LieGroupOps.to_tangent(v, epsilon) for v in a])

    @staticmethod  # type: ignore
    def storage_D_tangent(a):
        # type: (Element) -> geo.Matrix
        from symforce import geo

        mat = geo.Matrix(StorageOps.storage_dim(a), LieGroupOps.tangent_dim(a))
        s_inx = 0
        t_inx = 0
        for v in a:
            s_dim = StorageOps.storage_dim(v)
            t_dim = LieGroupOps.tangent_dim(v)
            mat[s_inx : s_inx + s_dim, t_inx : t_inx + t_dim] = LieGroupOps.storage_D_tangent(v)
            s_inx += s_dim
            t_inx += t_dim
        return mat
