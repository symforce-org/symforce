# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from .camera_cal import CameraCal
from .linear_camera_cal import LinearCameraCal

from symforce import geo
from symforce import sympy as sm
from symforce import typing as T


class EquirectangularCameraCal(CameraCal):
    """
    Equirectangular camera model with parameters [fx, fy, cx, cy].
    (fx, fy) representing focal length; (cx, cy) representing principal point.
    """

    NUM_DISTORTION_COEFFS = 0

    def pixel_from_camera_point(
        self, point: geo.V3, epsilon: T.Scalar = sm.epsilon()
    ) -> T.Tuple[geo.V2, T.Scalar]:
        cam_xz_norm = geo.V2(point[0], point[2]).norm(epsilon)
        ud_x = sm.atan2(point[0], point[2], epsilon=epsilon)
        ud_y = sm.atan2(point[1], cam_xz_norm, epsilon=0)

        linear_camera_cal = LinearCameraCal(
            self.focal_length.to_flat_list(), self.principal_point.to_flat_list()
        )
        pixel = linear_camera_cal.pixel_from_unit_depth(geo.V2(ud_x, ud_y))

        is_valid = sm.is_positive(point.squared_norm())
        return pixel, is_valid

    def camera_ray_from_pixel(
        self, pixel: geo.V2, epsilon: T.Scalar = sm.epsilon()
    ) -> T.Tuple[geo.V3, T.Scalar]:
        linear_camera_cal = LinearCameraCal(
            self.focal_length.to_flat_list(), self.principal_point.to_flat_list()
        )
        unit_depth = linear_camera_cal.unit_depth_from_pixel(pixel)
        xyz = geo.V3(
            sm.cos(unit_depth[1]) * sm.sin(unit_depth[0]),
            sm.sin(unit_depth[1]),
            sm.cos(unit_depth[1]) * sm.cos(unit_depth[0]),
        )

        is_valid = sm.logical_and(
            sm.is_positive(sm.pi - sm.Abs(unit_depth[0])),
            sm.is_positive(sm.pi / 2 - sm.Abs(unit_depth[1])),
            unsafe=True,
        )
        return xyz, is_valid
