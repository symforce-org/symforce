# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce.internal.symbolic as sf
from symforce import geo
from symforce import typing as T

from .camera_cal import CameraCal


class LinearCameraCal(CameraCal):
    """
    Standard pinhole camera w/ four parameters [fx, fy, cx, cy].
    (fx, fy) representing focal length; (cx, cy) representing principal point.
    """

    NUM_DISTORTION_COEFFS = 0

    @staticmethod
    def project(point: geo.V3, epsilon: T.Scalar = sf.epsilon()) -> T.Tuple[geo.V2, T.Scalar]:
        """
        Linearly project the 3D point by dividing by the depth.

        Points behind the camera (z <= 0 in the camera frame) are marked as invalid.

        Args:
            point: 3D point

        Returns:
            value_if_is_valid: Result of projection if the point is valid
            is_valid: 1 if the point is valid; 0 otherwise
        """
        x = point[0]
        y = point[1]

        # TODO (nathan): Remove if-statement?
        if epsilon == 0:
            z = point[2]
        else:
            z = sf.Max(point[2], epsilon)

        return geo.Vector2(x / z, y / z), sf.is_positive(point[2])

    def pixel_from_unit_depth(self, unit_depth_coords: geo.V2) -> geo.V2:
        """
        Convert point in unit-depth image plane to pixel coords by applying camera matrix.
        """
        return geo.Vector2(
            unit_depth_coords[0] * self.focal_length[0] + self.principal_point[0],
            unit_depth_coords[1] * self.focal_length[1] + self.principal_point[1],
        )

    def unit_depth_from_pixel(self, pixel: geo.V2) -> geo.V2:
        """
        Convert point in pixel coordinates to unit-depth image plane by applying K_inv.
        """
        return geo.Vector2(
            (pixel[0] - self.principal_point[0]) / self.focal_length[0],
            (pixel[1] - self.principal_point[1]) / self.focal_length[1],
        )

    def pixel_from_camera_point(
        self, point: geo.V3, epsilon: T.Scalar = sf.epsilon()
    ) -> T.Tuple[geo.V2, T.Scalar]:
        unit_depth, is_valid = LinearCameraCal.project(point, epsilon=epsilon)
        return self.pixel_from_unit_depth(unit_depth), is_valid

    def camera_ray_from_pixel(
        self, pixel: geo.V2, epsilon: T.Scalar = sf.epsilon()
    ) -> T.Tuple[geo.V3, T.Scalar]:
        unit_depth = self.unit_depth_from_pixel(pixel)
        camera_ray = geo.Vector3(unit_depth[0], unit_depth[1], 1)
        is_valid = sf.S.One
        return camera_ray, is_valid
