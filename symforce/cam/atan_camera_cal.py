# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import symforce.internal.symbolic as sf
from symforce import geo
from symforce import typing as T

from .camera_cal import CameraCal
from .linear_camera_cal import LinearCameraCal


class ATANCameraCal(CameraCal):
    """
    ATAN camera with 5 parameters [fx, fy, cx, cy, omega].
    (fx, fy) representing focal length, (cx, cy) representing principal point,
    and omega representing the distortion parameter.

    See here for more details:
    https://hal.inria.fr/inria-00267247/file/distcalib.pdf
    """

    NUM_DISTORTION_COEFFS = 1

    def __init__(
        self,
        focal_length: T.Sequence[T.Scalar],
        principal_point: T.Sequence[T.Scalar],
        omega: T.Scalar,
    ) -> None:
        super().__init__(focal_length, principal_point, [omega])

    @property
    def omega(self) -> T.Scalar:
        return self.distortion_coeffs[0]

    @omega.setter
    def omega(self, value: T.Scalar) -> None:
        self.distortion_coeffs[0] = value

    @classmethod
    def symbolic(cls, name: str, **kwargs: T.Any) -> ATANCameraCal:
        with sf.scope(name):
            return cls(
                focal_length=sf.symbols("f_x f_y"),
                principal_point=sf.symbols("c_x c_y"),
                omega=sf.Symbol("omega"),
            )

    @classmethod
    def storage_order(cls) -> T.Tuple[T.Tuple[str, int], ...]:
        return ("focal_length", 2), ("principal_point", 2), ("omega", 1)

    def pixel_from_camera_point(
        self, point: geo.V3, epsilon: T.Scalar = sf.epsilon()
    ) -> T.Tuple[geo.V2, T.Scalar]:

        # Compute undistorted point in unit depth image plane
        linear_camera_cal = LinearCameraCal(
            self.focal_length.to_flat_list(), self.principal_point.to_flat_list()
        )
        unit_depth, is_valid = linear_camera_cal.project(point, epsilon)

        # Compute distortion weight
        omega = self.distortion_coeffs[0]
        undistorted_radius = unit_depth.norm(epsilon)
        distortion_weight = sf.atan(2 * undistorted_radius * sf.tan(omega / 2.0)) / (
            undistorted_radius * omega
        )

        # Apply distortion weight and convert to pixels
        distorted_unit_depth_coords = unit_depth * distortion_weight
        pixel = linear_camera_cal.pixel_from_unit_depth(distorted_unit_depth_coords)
        return pixel, is_valid

    def camera_ray_from_pixel(
        self, pixel: geo.V2, epsilon: T.Scalar = sf.epsilon()
    ) -> T.Tuple[geo.V3, T.Scalar]:

        # Compute distorted unit depth coords
        linear_camera_cal = LinearCameraCal(
            self.focal_length.to_flat_list(), self.principal_point.to_flat_list()
        )
        distorted_unit_depth_coords = linear_camera_cal.unit_depth_from_pixel(pixel)

        # Compute undistortion weight
        omega = self.distortion_coeffs[0]
        distorted_radius = distorted_unit_depth_coords.norm(epsilon)
        undistortion_weight = sf.tan(distorted_radius * omega) / (
            2 * distorted_radius * sf.tan(omega / 2.0)
        )

        # Apply weight and convert to camera ray
        unit_depth = undistortion_weight * distorted_unit_depth_coords
        camera_ray = geo.V3(unit_depth[0], unit_depth[1], 1)
        is_valid = sf.Max(sf.sign(sf.pi / 2 - sf.Abs(distorted_radius * omega)), 0)
        return camera_ray, is_valid
