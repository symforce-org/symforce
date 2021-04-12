import math
import numpy
import typing as T

import sym  # pylint: disable=unused-import


class LieGroupOps(object):
    """
    Python LieGroupOps implementation for <class 'symforce.geo.rot3.Rot3'>.
    """

    @staticmethod
    def from_tangent(vec, epsilon):
        # type: (T.Sequence[float], float) -> T.List[float]

        # Total ops: 16

        # Input arrays

        # Intermediate terms (3)
        _tmp0 = math.sqrt(epsilon ** 2 + vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
        _tmp1 = (1.0 / 2.0) * _tmp0
        _tmp2 = math.sin(_tmp1) / _tmp0

        # Output terms
        _res = [0.0] * 4
        _res[0] = _tmp2 * vec[0]
        _res[1] = _tmp2 * vec[1]
        _res[2] = _tmp2 * vec[2]
        _res[3] = math.cos(_tmp1)
        return _res

    @staticmethod
    def to_tangent(a, epsilon):
        # type: (sym.Rot3, float) -> T.List[float]

        # Total ops: 19

        # Input arrays
        _a = a.data

        # Intermediate terms (2)
        _tmp0 = min(abs(_a[3]), 1 - epsilon)
        _tmp1 = (
            2
            * (2 * min(0, (0.0 if _a[3] == 0 else math.copysign(1, _a[3]))) + 1)
            * math.acos(_tmp0)
            / math.sqrt(1 - _tmp0 ** 2)
        )

        # Output terms
        _res = [0.0] * 3
        _res[0] = _a[0] * _tmp1
        _res[1] = _a[1] * _tmp1
        _res[2] = _a[2] * _tmp1
        return _res

    @staticmethod
    def retract(a, vec, epsilon):
        # type: (sym.Rot3, T.Sequence[float], float) -> T.List[float]

        # Total ops: 49

        # Input arrays
        _a = a.data

        # Intermediate terms (8)
        _tmp0 = math.sqrt(epsilon ** 2 + vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
        _tmp1 = (1.0 / 2.0) * _tmp0
        _tmp2 = math.sin(_tmp1) / _tmp0
        _tmp3 = _a[1] * _tmp2
        _tmp4 = _a[2] * _tmp2
        _tmp5 = math.cos(_tmp1)
        _tmp6 = _a[3] * _tmp2
        _tmp7 = _a[0] * _tmp2

        # Output terms
        _res = [0.0] * 4
        _res[0] = _a[0] * _tmp5 + _tmp3 * vec[2] - _tmp4 * vec[1] + _tmp6 * vec[0]
        _res[1] = _a[1] * _tmp5 + _tmp4 * vec[0] + _tmp6 * vec[1] - _tmp7 * vec[2]
        _res[2] = _a[2] * _tmp5 - _tmp3 * vec[0] + _tmp6 * vec[2] + _tmp7 * vec[1]
        _res[3] = _a[3] * _tmp5 - _tmp3 * vec[1] - _tmp4 * vec[2] - _tmp7 * vec[0]
        return _res

    @staticmethod
    def local_coordinates(a, b, epsilon):
        # type: (sym.Rot3, sym.Rot3, float) -> T.List[float]

        # Total ops: 57

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (3)
        _tmp0 = _a[0] * _b[0] + _a[1] * _b[1] + _a[2] * _b[2] + _a[3] * _b[3]
        _tmp1 = min(abs(_tmp0), 1 - epsilon)
        _tmp2 = (
            2
            * (2 * min(0, (0.0 if _tmp0 == 0 else math.copysign(1, _tmp0))) + 1)
            * math.acos(_tmp1)
            / math.sqrt(1 - _tmp1 ** 2)
        )

        # Output terms
        _res = [0.0] * 3
        _res[0] = _tmp2 * (-_a[0] * _b[3] - _a[1] * _b[2] + _a[2] * _b[1] + _a[3] * _b[0])
        _res[1] = _tmp2 * (_a[0] * _b[2] - _a[1] * _b[3] - _a[2] * _b[0] + _a[3] * _b[1])
        _res[2] = _tmp2 * (-_a[0] * _b[1] + _a[1] * _b[0] - _a[2] * _b[3] + _a[3] * _b[2])
        return _res
