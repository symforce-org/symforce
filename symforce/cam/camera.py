# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce.internal.symbolic as sf
from symforce import geo
from symforce import typing as T

from .camera_cal import CameraCal


class Camera:
    """
    Camera with a given camera calibration and an optionally specified image size (width, height).
    If the image size is specified, we use it to check whether pixels (either given or computed by
    projection of 3D points into the image frame) are in the image frame and thus valid/invalid.
    """

    # Type that represents this or any subclasses
    CameraT = T.TypeVar("CameraT", bound="Camera")

    def __init__(self, calibration: CameraCal, image_size: T.Sequence[T.Scalar] = None) -> None:
        self.calibration = calibration

        if image_size is not None:
            assert len(image_size) == 2
            self.image_size: T.Optional[geo.V2] = geo.V2(image_size)
        else:
            self.image_size = None

    @property
    def focal_length(self) -> geo.V2:
        return self.calibration.focal_length

    @property
    def principal_point(self) -> geo.V2:
        return self.calibration.principal_point

    @property
    def distortion_coeffs(self) -> geo.Matrix:
        return self.calibration.distortion_coeffs

    def __repr__(self) -> str:
        if self.image_size:
            image_size_str = repr(self.image_size.to_storage())
        else:
            image_size_str = "None"
        return "<{}\n  CameraCal={}\n  image_size={}>".format(
            self.__class__.__name__, repr(self.calibration), image_size_str
        )

    def pixel_from_camera_point(
        self, point: geo.V3, epsilon: T.Scalar = sf.epsilon()
    ) -> T.Tuple[geo.V2, T.Scalar]:
        """
        Project a 3D point in the camera frame into 2D pixel coordinates.

        Return:
            pixel: (x, y) coordinate in pixels if valid
            is_valid: 1 if the operation is within bounds (including image_size bounds) else 0
        """
        pixel, is_valid = self.calibration.pixel_from_camera_point(point, epsilon)
        is_valid *= self.maybe_check_in_view(pixel)
        return pixel, is_valid

    def camera_ray_from_pixel(
        self, pixel: geo.V2, epsilon: T.Scalar = sf.epsilon(), normalize: bool = False
    ) -> T.Tuple[geo.V3, T.Scalar]:
        """
        Backproject a 2D pixel coordinate into a 3D ray in the camera frame.

        NOTE: If image_size is specified and the given pixel is out of
        bounds, is_valid will be set to zero.

        Args:
            normalize: Whether camera_ray will be normalized (False by default)

        Return:
            camera_ray: The ray in the camera frame
            is_valid: 1 if the operation is within bounds else 0
        """
        camera_ray, is_valid = self.calibration.camera_ray_from_pixel(pixel, epsilon)

        if normalize:
            camera_ray = camera_ray.normalized(epsilon=epsilon)

        is_valid *= self.maybe_check_in_view(pixel)
        return camera_ray, is_valid

    def has_camera_ray_from_pixel(self) -> bool:
        """
        Returns True if self has implemented the method camera_ray_from_pixel, and False
        otherwise.
        """
        return self.calibration.has_camera_ray_from_pixel()

    def maybe_check_in_view(self, pixel: geo.V2) -> int:
        if self.image_size is None:
            return sf.S.One

        return self.in_view(pixel, self.image_size)

    @staticmethod
    def in_view(pixel: geo.V2, image_size: geo.V2) -> int:
        """
        Returns 1.0 if the pixel coords are in bounds of the image, 0.0 otherwise.
        """
        return sf.Mul(
            *[
                sf.Max(0, sf.sign(bound - value - sf.S.One) * sf.sign(value))
                for bound, value in zip(image_size.to_flat_list(), pixel.to_flat_list())
            ]
        )

    def subs(self: CameraT, *args: T.Any, **kwargs: T.Any) -> CameraT:
        """
        Substitute given values of each scalar element into a new instance.
        """
        return self.__class__(
            calibration=self.calibration.subs(*args, **kwargs),
            image_size=None
            if self.image_size is None
            else self.image_size.subs(*args, **kwargs).to_flat_list(),
        )
