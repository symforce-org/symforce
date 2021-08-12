import math
import numpy
import typing as T

import sym  # pylint: disable=unused-import


class GroupOps(object):
    """
    Python GroupOps implementation for <class 'symforce.geo.rot2.Rot2'>.
    """

    @staticmethod
    def identity():
        # type: () -> T.List[float]

        # Total ops: 0

        # Input arrays

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 2
        _res[0] = 1
        _res[1] = 0
        return _res

    @staticmethod
    def inverse(a):
        # type: (sym.Rot2) -> T.List[float]

        # Total ops: 1

        # Input arrays
        _a = a.data

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 2
        _res[0] = _a[0]
        _res[1] = -_a[1]
        return _res

    @staticmethod
    def compose(a, b):
        # type: (sym.Rot2, sym.Rot2) -> T.List[float]

        # Total ops: 7

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 2
        _res[0] = _a[0] * _b[0] - _a[1] * _b[1]
        _res[1] = _a[0] * _b[1] + _a[1] * _b[0]
        return _res

    @staticmethod
    def between(a, b):
        # type: (sym.Rot2, sym.Rot2) -> T.List[float]

        # Total ops: 7

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 2
        _res[0] = _a[0] * _b[0] + _a[1] * _b[1]
        _res[1] = _a[0] * _b[1] - _a[1] * _b[0]
        return _res

    @staticmethod
    def inverse_with_jacobian(a):
        # type: (sym.Rot2) -> T.Tuple[T.List[float], T.List[float]]

        # Total ops: 6

        # Input arrays
        _a = a.data

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 2
        _res[0] = _a[0]
        _res[1] = -_a[1]
        _res_D_a = [0.0] * 1
        _res_D_a[0] = -_a[0] ** 2 - _a[1] ** 2
        return _res, _res_D_a

    @staticmethod
    def compose_with_jacobians(a, b):
        # type: (sym.Rot2, sym.Rot2) -> T.Tuple[T.List[float], T.List[float], T.List[float]]

        # Total ops: 29

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (2)
        _tmp0 = _a[0] * _b[0] - _a[1] * _b[1]
        _tmp1 = _a[0] * _b[1] + _a[1] * _b[0]

        # Output terms
        _res = [0.0] * 2
        _res[0] = _tmp0
        _res[1] = _tmp1
        _res_D_a = [0.0] * 1
        _res_D_a[0] = _a[0] * (_b[0] * _tmp0 + _b[1] * _tmp1) - _a[1] * (
            -_b[0] * _tmp1 + _b[1] * _tmp0
        )
        _res_D_b = [0.0] * 1
        _res_D_b[0] = _b[0] * (_a[0] * _tmp0 + _a[1] * _tmp1) - _b[1] * (
            -_a[0] * _tmp1 + _a[1] * _tmp0
        )
        return _res, _res_D_a, _res_D_b

    @staticmethod
    def between_with_jacobians(a, b):
        # type: (sym.Rot2, sym.Rot2) -> T.Tuple[T.List[float], T.List[float], T.List[float]]

        # Total ops: 33

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (2)
        _tmp0 = _a[0] * _b[0] + _a[1] * _b[1]
        _tmp1 = _a[0] * _b[1] - _a[1] * _b[0]

        # Output terms
        _res = [0.0] * 2
        _res[0] = _tmp0
        _res[1] = _tmp1
        _res_D_a = [0.0] * 1
        _res_D_a[0] = _a[0] * (-_b[0] * _tmp0 - _b[1] * _tmp1) - _a[1] * (
            -_b[0] * _tmp1 + _b[1] * _tmp0
        )
        _res_D_b = [0.0] * 1
        _res_D_b[0] = _b[0] * (_a[0] * _tmp0 - _a[1] * _tmp1) - _b[1] * (
            -_a[0] * _tmp1 - _a[1] * _tmp0
        )
        return _res, _res_D_a, _res_D_b
