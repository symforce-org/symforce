# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import symforce.internal.symbolic as sf
from symforce import geo
from symforce import typing as T

from .camera import Camera
from .camera_cal import CameraCal


class PosedCamera(Camera):
    """
    Camera with a given pose, camera calibration, and an optionally specified image size.
    If the image size is specified, we use it to check whether pixels (either given or computed by
    projection of 3D points into the image frame) are in the image frame and thus valid/invalid.
    """

    # Type that represents this or any subclasses
    PosedCameraT = T.TypeVar("PosedCameraT", bound="PosedCamera")

    def __init__(
        self, pose: geo.Pose3, calibration: CameraCal, image_size: T.Sequence[T.Scalar] = None
    ) -> None:
        super().__init__(calibration=calibration, image_size=image_size)
        self.pose = pose  # global_T_cam

    def __repr__(self) -> str:
        return "<{}\n  Pose={}\n  Camera={}>".format(
            self.__class__.__name__, self.pose.__repr__(), super().__repr__()
        )

    def pixel_from_global_point(
        self, point: geo.Vector3, epsilon: T.Scalar = sf.epsilon()
    ) -> T.Tuple[geo.Vector2, T.Scalar]:
        """
        Transforms the given point into the camera frame using the given camera pose and then
        uses the given camera calibration to compute the resulted pixel coordinates of the
        projected point.

        Args:
            point: Vector written in camera frame.
            epsilon: Small value intended to prevent division by 0.

        Return:
            pixel: UV coodinates in pixel units, assuming the point is in view
            is_valid: 1 if point is valid
        """
        camera_point = self.pose.inverse() * point
        pixel, is_valid = self.pixel_from_camera_point(camera_point, epsilon)
        return pixel, is_valid

    def global_point_from_pixel(
        self, pixel: geo.Vector2, range_to_point: T.Scalar, epsilon: T.Scalar = sf.epsilon()
    ) -> T.Tuple[geo.Vector3, T.Scalar]:
        """
        Computes a point written in the global frame along the ray passing through the center
        of the given pixel. The point is positioned at a given range along the ray.

        Args:
            pixel: Vector in pixels in camera image.
            range_to_point: Distance of the returned point along the ray passing through pixel
            epsilon: Small value intended to prevent division by 0.

        Return:
            global_point: The point in the global frame.
            is_valid: 1 if point is valid
        """
        # ray out from the world camera position in the global frame
        camera_ray, is_valid = self.camera_ray_from_pixel(pixel, epsilon)
        camera_point = (camera_ray / camera_ray.norm(epsilon=epsilon)) * range_to_point
        global_point = self.pose * camera_point
        return global_point, is_valid

    def warp_pixel(
        self,
        pixel: geo.Vector2,
        inverse_range: T.Scalar,
        target_cam: PosedCamera,
        epsilon: T.Scalar = sf.epsilon(),
    ) -> T.Tuple[geo.Vector2, T.Scalar]:
        """
        Project a pixel in this camera into another camera.

        Args:
            pixel: Pixel in the source camera
            inverse_range: Inverse distance along the ray to the global point
            target_cam: Camera to project global point into

        Return:
            pixel: Pixel in the target camera
            is_valid: 1 if given point is valid in source camera and target camera
        """
        # NOTE(ryan): let me explain the math here since we're not doing the most
        # obvious implementation. The math can be simplified by taking advantage of
        # the fact that we're projecting into camera 1, and so the point in camera 1's
        # frame can be scaled arbitrarily and it will still project into the same pixel.
        # The idea is as follows. Let p be the unit ray in camera 0's frame, [R, t] be the
        # transform between the two camera frames. The point in camera 1's frame is found by
        # projecting out at the given range and transforming: R*[p/inv_range] + t.
        # We can now scale this arbitrarily and it will project to the same pixel in camera 1;
        # let's multiply it by inv_range to get: R*p + t*inv_range. This is the point that
        # we project into camera1. Note that this avoids dividing by inv_range, so inv_range
        # can be == 0 without a special case.

        # Project out to a unit ray.
        camera_ray, is_valid_point = self.camera_ray_from_pixel(pixel, epsilon)
        camera_point = camera_ray / camera_ray.norm()

        # Transform into the other camera at this inverse range.
        # NOTE(ryan): expand out the math here, since grouping (R0*R1)*p is more operations
        # than R0*(R1*p).
        transformed_point = target_cam.pose.R.inverse() * (
            self.pose.R * camera_point + (self.pose.t - target_cam.pose.t) * inverse_range
        )

        # Project into the target camera.
        pixel, is_valid_projection = target_cam.pixel_from_camera_point(
            transformed_point, epsilon=epsilon
        )

        return pixel, is_valid_point * is_valid_projection

    def subs(self: PosedCameraT, *args: T.Any, **kwargs: T.Any) -> PosedCameraT:
        """
        Substitute given values of each scalar element into a new instance.
        """
        return self.__class__(
            pose=self.pose.subs(*args, **kwargs),
            calibration=self.calibration.subs(*args, **kwargs),
            image_size=None
            if self.image_size is None
            else self.image_size.subs(*args, **kwargs).to_flat_list(),
        )
