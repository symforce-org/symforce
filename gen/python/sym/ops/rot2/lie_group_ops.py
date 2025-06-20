# -----------------------------------------------------------------------------
# This file was autogenerated by symforce from template:
#     ops/CLASS/lie_group_ops.py.jinja
# Do NOT modify by hand.
# -----------------------------------------------------------------------------

# ruff: noqa: PLR0915, F401, PLW0211, PLR0914

import math
import typing as T

import numpy

import sym


class LieGroupOps(object):
    """
    Python LieGroupOps implementation for :py:class:`symforce.geo.rot2.Rot2`.
    """

    @staticmethod
    def from_tangent(vec, epsilon):
        # type: (numpy.ndarray, float) -> sym.Rot2

        # Total ops: 2

        # Input arrays
        if vec.shape == (1,):
            vec = vec.reshape((1, 1))
        elif vec.shape != (1, 1):
            raise IndexError(
                "vec is expected to have shape (1, 1) or (1,); instead had shape {}".format(
                    vec.shape
                )
            )

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 2
        _res[0] = math.cos(vec[0, 0])
        _res[1] = math.sin(vec[0, 0])
        return sym.Rot2.from_storage(_res)

    @staticmethod
    def to_tangent(a, epsilon):
        # type: (sym.Rot2, float) -> numpy.ndarray

        # Total ops: 3

        # Input arrays
        _a = a.data

        # Intermediate terms (0)

        # Output terms
        _res = numpy.zeros(1)
        _res[0] = math.atan2(_a[1], _a[0] + math.copysign(epsilon, _a[0]))
        return _res

    @staticmethod
    def retract(a, vec, epsilon):
        # type: (sym.Rot2, numpy.ndarray, float) -> sym.Rot2

        # Total ops: 8

        # Input arrays
        _a = a.data
        if vec.shape == (1,):
            vec = vec.reshape((1, 1))
        elif vec.shape != (1, 1):
            raise IndexError(
                "vec is expected to have shape (1, 1) or (1,); instead had shape {}".format(
                    vec.shape
                )
            )

        # Intermediate terms (2)
        _tmp0 = math.cos(vec[0, 0])
        _tmp1 = math.sin(vec[0, 0])

        # Output terms
        _res = [0.0] * 2
        _res[0] = _a[0] * _tmp0 - _a[1] * _tmp1
        _res[1] = _a[0] * _tmp1 + _a[1] * _tmp0
        return sym.Rot2.from_storage(_res)

    @staticmethod
    def local_coordinates(a, b, epsilon):
        # type: (sym.Rot2, sym.Rot2, float) -> numpy.ndarray

        # Total ops: 9

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (1)
        _tmp0 = _a[0] * _b[0] + _a[1] * _b[1]

        # Output terms
        _res = numpy.zeros(1)
        _res[0] = math.atan2(_a[0] * _b[1] - _a[1] * _b[0], _tmp0 + math.copysign(epsilon, _tmp0))
        return _res

    @staticmethod
    def interpolate(a, b, alpha, epsilon):
        # type: (sym.Rot2, sym.Rot2, float, float) -> sym.Rot2

        # Total ops: 18

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (4)
        _tmp0 = _a[0] * _b[0] + _a[1] * _b[1]
        _tmp1 = alpha * math.atan2(
            _a[0] * _b[1] - _a[1] * _b[0], _tmp0 + math.copysign(epsilon, _tmp0)
        )
        _tmp2 = math.cos(_tmp1)
        _tmp3 = math.sin(_tmp1)

        # Output terms
        _res = [0.0] * 2
        _res[0] = _a[0] * _tmp2 - _a[1] * _tmp3
        _res[1] = _a[0] * _tmp3 + _a[1] * _tmp2
        return sym.Rot2.from_storage(_res)
