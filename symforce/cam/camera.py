from .camera_cal import CameraCal

from symforce import geo
from symforce import sympy as sm
from symforce import types as T


class Camera(object):
    """
    Camera with a given camera calibration and an optionally specified image size.
    If the image size is specified, we use it to check whether pixels (either given or computed by
    projection of 3D points into the image frame) are in the image frame and thus valid/invalid.
    """

    def __init__(self, calibration, image_size=None):
        # type: (CameraCal, T.Sequence[int]) -> None
        self.calibration = calibration

        if image_size is not None:
            assert len(image_size) == 2
            self.image_size = geo.V2(image_size)
        else:
            self.image_size = None  # type: ignore

    @property
    def focal_length(self):
        # type: () -> geo.Matrix21
        return self.calibration.focal_length

    @property
    def principal_point(self):
        # type: () -> geo.Matrix21
        return self.calibration.principal_point

    @property
    def distortion_coeffs(self):
        # type: () -> geo.Matrix
        return self.calibration.distortion_coeffs

    def __repr__(self):
        # type: () -> str
        if self.image_size:
            image_size_str = repr(self.image_size.to_storage())
        else:
            image_size_str = "None"
        return "<{}\n  CameraCal={}\n  image_size={}>".format(
            self.__class__.__name__, repr(self.calibration), image_size_str
        )

    def pixel_from_camera_point(self, point, epsilon=0):
        # type: (geo.Matrix31, T.Scalar) -> T.Tuple[geo.Matrix21, T.Scalar]
        """
        Project a 3D point in the camera frame into 2D pixel coordinates.

        Return:
            pixel: (x, y) coordinate in pixels if valid
            is_valid: 1 if the operation is within bounds (including image_size bounds) else 0
        """
        pixel, is_valid = self.calibration.pixel_from_camera_point(point, epsilon)
        is_valid *= self.maybe_check_in_view(pixel)
        return pixel, is_valid

    def camera_ray_from_pixel(self, pixel, epsilon=0):
        # type: (geo.Matrix21, T.Scalar) -> T.Tuple[geo.Matrix31, T.Scalar]
        """
        Backproject a 2D pixel coordinate into a 3D ray in the camera frame.

        NOTE: If image_size is specified and the given pixel is out of
        bounds, is_valid will be set to zero.

        Return:
            camera_ray: The ray in the camera frame (NOT normalized)
            is_valid: 1 if the operation is within bounds else 0
        """
        camera_ray, is_valid = self.calibration.camera_ray_from_pixel(pixel, epsilon)
        is_valid *= self.maybe_check_in_view(pixel)
        return camera_ray, is_valid

    def maybe_check_in_view(self, pixel):
        # type: (geo.Matrix21) -> int
        if not self.image_size:
            return sm.S.One

        return self.in_view(pixel, self.image_size)

    @staticmethod
    def in_view(pixel, image_size):
        # type: (geo.Matrix21, geo.Matrix21) -> int
        """
        Returns 1.0 if the pixel coords are in bounds of the image, 0.0 otherwise.
        """
        return sm.Mul(
            *[
                sm.Max(0, sm.sign(bound - value - sm.S.One) * sm.sign(value))
                for bound, value in zip(image_size.to_flat_list(), pixel.to_flat_list())
            ]
        )
