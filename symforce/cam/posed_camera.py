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

    def pixel_from_global_point(self, point, epsilon=0):
        # type: (geo.Matrix31, T.Scalar) -> T.Tuple[geo.Matrix21, T.Scalar]
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

    def global_point_from_pixel(self, pixel, range_to_point, epsilon=0):
        # type: (geo.Matrix21, T.Scalar, T.Scalar) -> T.Tuple[geo.Matrix31, T.Scalar]
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

    def warp_pixel(self, pixel, inverse_range, target_cam, epsilon=0):
        # type: (geo.Matrix21, T.Scalar, PosedCamera, T.Scalar) -> T.Tuple[geo.Matrix21, T.Scalar]
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
        if inverse_range == 0:
            # Point is at infinity, use rotations to avoid NaNs.
            camera_ray_self, is_valid_ray = self.camera_ray_from_pixel(pixel, epsilon)
            # camera_ray_target is a ray in the target cam that is parallel to camera_ray_self,
            # and is written in target cam coordinates (two parallel rays intersect at infinity)
            camera_ray_target = target_cam.pose.R.inverse() * (self.pose.R * camera_ray_self)
            pixel, is_valid_projection = target_cam.pixel_from_camera_point(camera_ray_target)
            return pixel, is_valid_ray * is_valid_projection

        global_point, is_valid_point = self.global_point_from_pixel(
            pixel, sm.S.One / (inverse_range + epsilon), epsilon
        )
        pixel, is_valid_projection = target_cam.pixel_from_global_point(global_point, epsilon)

        return pixel, is_valid_point * is_valid_projection
