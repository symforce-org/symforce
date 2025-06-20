# -----------------------------------------------------------------------------
# This file was autogenerated by symforce from template:
#     function/FUNCTION.py.jinja
# Do NOT modify by hand.
# -----------------------------------------------------------------------------

# ruff: noqa: F401, PLR0912, PLR0913, PLR0914, PLR0915, PLR0917, RUF100

import math
import typing as T

import numpy

import sym


def az_el_from_point(nav_T_cam, nav_t_point, epsilon):
    # type: (sym.Pose3, numpy.ndarray, float) -> numpy.ndarray
    """
    Transform a nav point into azimuth / elevation angles in the
    camera frame.

    Args:
        nav_T_cam (sf.Pose3): camera pose in the world
        nav_t_point (sf.Matrix): nav point
        epsilon (Scalar): small number to avoid singularities

    Returns:
        sf.Matrix: (azimuth, elevation)
    """

    # Total ops: 75

    # Input arrays
    _nav_T_cam = nav_T_cam.data
    if nav_t_point.shape == (3,):
        nav_t_point = nav_t_point.reshape((3, 1))
    elif nav_t_point.shape != (3, 1):
        raise IndexError(
            "nav_t_point is expected to have shape (3, 1) or (3,); instead had shape {}".format(
                nav_t_point.shape
            )
        )

    # Intermediate terms (23)
    _tmp0 = 2 * _nav_T_cam[0] * _nav_T_cam[3]
    _tmp1 = 2 * _nav_T_cam[2]
    _tmp2 = _nav_T_cam[1] * _tmp1
    _tmp3 = _tmp0 + _tmp2
    _tmp4 = -2 * _nav_T_cam[2] ** 2
    _tmp5 = 1 - 2 * _nav_T_cam[0] ** 2
    _tmp6 = _tmp4 + _tmp5
    _tmp7 = 2 * _nav_T_cam[1]
    _tmp8 = _nav_T_cam[0] * _tmp7
    _tmp9 = _nav_T_cam[3] * _tmp1
    _tmp10 = _tmp8 - _tmp9
    _tmp11 = (
        -_nav_T_cam[4] * _tmp10
        - _nav_T_cam[5] * _tmp6
        - _nav_T_cam[6] * _tmp3
        + _tmp10 * nav_t_point[0, 0]
        + _tmp3 * nav_t_point[2, 0]
        + _tmp6 * nav_t_point[1, 0]
    )
    _tmp12 = -2 * _nav_T_cam[1] ** 2
    _tmp13 = _tmp12 + _tmp4 + 1
    _tmp14 = _tmp8 + _tmp9
    _tmp15 = _nav_T_cam[3] * _tmp7
    _tmp16 = _nav_T_cam[0] * _tmp1
    _tmp17 = -_tmp15 + _tmp16
    _tmp18 = (
        -_nav_T_cam[4] * _tmp13
        - _nav_T_cam[5] * _tmp14
        - _nav_T_cam[6] * _tmp17
        + _tmp13 * nav_t_point[0, 0]
        + _tmp14 * nav_t_point[1, 0]
        + _tmp17 * nav_t_point[2, 0]
    )
    _tmp19 = -_tmp0 + _tmp2
    _tmp20 = _tmp12 + _tmp5
    _tmp21 = _tmp15 + _tmp16
    _tmp22 = (
        -_nav_T_cam[4] * _tmp21
        - _nav_T_cam[5] * _tmp19
        - _nav_T_cam[6] * _tmp20
        + _tmp19 * nav_t_point[1, 0]
        + _tmp20 * nav_t_point[2, 0]
        + _tmp21 * nav_t_point[0, 0]
    )

    # Output terms
    _res = numpy.zeros(2)
    _res[0] = math.atan2(_tmp11, _tmp18 + math.copysign(epsilon, _tmp18))
    _res[1] = (
        -math.acos(_tmp22 / math.sqrt(_tmp11**2 + _tmp18**2 + _tmp22**2 + epsilon))
        + (1.0 / 2.0) * math.pi
    )
    return _res
