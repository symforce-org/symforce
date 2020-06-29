from .camera_cal import CameraCal
from .camera import Camera

from symforce import geo
from symforce import sympy as sm
from symforce import types as T
from symforce.python_util import classproperty


class PosedCamera(Camera):
    """
    Camera with a given pose, camera calibration, and an optionally specified image size.
    If the image size is specified, we use it to check whether pixels (either given or computed by
    projection of 3D points into the image frame) are in the image frame and thus valid/invalid.
    """

    def __init__(self, pose, calibration, image_size=None):
        # type: (geo.Pose3, CameraCal, T.Sequence[int]) -> None
        super(PosedCamera, self).__init__(calibration=calibration, image_size=image_size)
        self.pose = pose  # global_T_cam

    def __repr__(self):
        # type: () -> str
        return "<{}\n  Pose={}\n  Camera={}>".format(
            self.__class__.__name__, self.pose.__repr__(), super(PosedCamera, self).__repr__()
        )

    def pixel_coords_from_global_point(self, point, epsilon=0):
        # type: (geo.Matrix31, T.Scalar) -> T.Tuple[geo.Matrix21, T.Scalar]
        """
        Transforms the given point into the camera frame using the given camera pose and then
        uses the given camera calibration to compute the resulted pixel coordinates of the
        projected point.

        Args:
            point: Vector written in camera frame.
            epsilon: Small value intended to prevent division by 0.

        Return:
            pixel_coords: UV coodinates in pixel units, assuming the point is in view
            is_valid: 1 if point is valid
        """
        camera_point = self.pose.inverse() * point
        pixel_coords, is_valid = self.pixel_coords_from_camera_point(camera_point, epsilon)
        return pixel_coords, is_valid

    def global_point_from_pixel_coords(self, pixel_coords, range_to_point, epsilon=0):
        # type: (geo.Matrix21, T.Scalar, T.Scalar) -> T.Tuple[geo.Matrix31, T.Scalar]
        """
        Computes a point written in the global frame along the ray passing through the center
        of the given pixel. The point is positioned at a given range along the ray.

        Args:
            pixel_coords: Vector in pixels in camera image.
            range_to_point: Distance of the returned point along the ray passing through pixel_coords
            epsilon: Small value intended to prevent division by 0.

        Return:
            global_point: The point in the global frame.
            is_valid: 1 if point is valid
        """
        # ray out from the world camera position in the global frame
        camera_ray, is_valid = self.camera_ray_from_pixel_coords(pixel_coords, epsilon)
        camera_point = (camera_ray / camera_ray.norm(epsilon=epsilon)) * range_to_point
        global_point = self.pose * camera_point
        return global_point, is_valid
