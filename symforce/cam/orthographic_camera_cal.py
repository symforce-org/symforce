# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce.internal.symbolic as sf
from symforce import geo
from symforce import typing as T

from .camera_cal import CameraCal


class OrthographicCameraCal(CameraCal):
    """
    Orthographic camera model with four parameters [fx, fy, cx, cy].

    It would be possible to define orthographic cameras with only two parameters [fx, fy] but we
    keep the [cx, cy] parameters for consistency with the CameraCal interface.

    The orthographic camera model can be thought of as a special case of the LinearCameraCal model,
    where (x,y,z) in the camera frame projects to pixel (x * fx + cx, y * fy + cy).
    The z-coordinate of the point is ignored in the projection, except that only points with
    positive z-coordinates are considered valid.

    Because this is a noncentral camera model, the camera_ray_from_pixel function is not implemented.
    """

    NUM_DISTORTION_COEFFS = 0

    def pixel_from_camera_point(
        self, point: geo.V3, epsilon: T.Scalar = sf.epsilon()
    ) -> T.Tuple[geo.V2, T.Scalar]:
        is_valid = sf.is_positive(point[2])
        return geo.Vector2(
            point[0] * self.focal_length[0] + self.principal_point[0],
            point[1] * self.focal_length[1] + self.principal_point[1],
        ), is_valid
