import math
import numpy
import typing as T

import sym  # pylint: disable=unused-import


class LieGroupOps(object):
    """
    Python LieGroupOps implementation for <class 'symforce.geo.rot2.Rot2'>.
    """

    @staticmethod
    def from_tangent(vec, epsilon):
        # type: (T.Sequence[float], float) -> T.List[float]

        # Total ops: 2

        # Input arrays

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 2
        _res[0] = math.cos(vec[0])
        _res[1] = math.sin(vec[0])
        return _res

    @staticmethod
    def to_tangent(a, epsilon):
        # type: (sym.Rot2, float) -> T.List[float]

        # Total ops: 1

        # Input arrays
        _a = a.data

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 1
        _res[0] = math.atan2(_a[1], _a[0])
        return _res

    @staticmethod
    def retract(a, vec, epsilon):
        # type: (sym.Rot2, T.Sequence[float], float) -> T.List[float]

        # Total ops: 9

        # Input arrays
        _a = a.data

        # Intermediate terms (2)
        _tmp0 = math.sin(vec[0])
        _tmp1 = math.cos(vec[0])

        # Output terms
        _res = [0.0] * 2
        _res[0] = _a[0] * _tmp1 - _a[1] * _tmp0
        _res[1] = _a[0] * _tmp0 + _a[1] * _tmp1
        return _res

    @staticmethod
    def local_coordinates(a, b, epsilon):
        # type: (sym.Rot2, sym.Rot2, float) -> T.List[float]

        # Total ops: 14

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (3)
        _tmp0 = (_a[0] ** 2 + _a[1] ** 2) ** (-1)
        _tmp1 = _a[0] * _tmp0
        _tmp2 = _a[1] * _tmp0

        # Output terms
        _res = [0.0] * 1
        _res[0] = math.atan2(-_b[0] * _tmp2 + _b[1] * _tmp1, _b[0] * _tmp1 + _b[1] * _tmp2)
        return _res
