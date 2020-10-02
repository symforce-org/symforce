from .camera_cal import CameraCal
from .linear_camera_cal import LinearCameraCal

from symforce import geo
from symforce import sympy as sm
from symforce import types as T


class ATANCameraCal(CameraCal):
    """
    ATAN camera with 5 parameters [fx, fy, cx, cy, omega].
    (fx, fy) representing focal length, (cx, cy) representing principal point,
    and omega representing the distortion parameter.

    See here for more details:
    https://hal.inria.fr/inria-00267247/file/distcalib.pdf
    """

    NUM_DISTORTION_COEFFS = 1

    def pixel_from_camera_point(self, point, epsilon=0):
        # type: (geo.Matrix31, T.Scalar) -> T.Tuple[geo.Matrix21, T.Scalar]

        # Compute undistorted point in unit depth image plane
        linear_camera_cal = LinearCameraCal(
            self.focal_length.to_flat_list(), self.principal_point.to_flat_list()
        )
        unit_depth, is_valid = linear_camera_cal.project(point, epsilon)

        # Compute distortion weight
        omega = self.distortion_coeffs[0]
        undistorted_radius = unit_depth.norm(epsilon)
        distortion_weight = sm.atan(2 * undistorted_radius * sm.tan(omega / 2.0)) / (
            undistorted_radius * omega
        )

        # Apply distortion weight and convert to pixels
        distorted_unit_depth_coords = unit_depth * distortion_weight
        pixel = linear_camera_cal.pixel_from_unit_depth(distorted_unit_depth_coords)
        return pixel, is_valid

    def camera_ray_from_pixel(self, pixel, epsilon=0):
        # type: (geo.Matrix21, T.Scalar) -> T.Tuple[geo.Matrix31, T.Scalar]

        # Compute distorted unit depth coords
        linear_camera_cal = LinearCameraCal(
            self.focal_length.to_flat_list(), self.principal_point.to_flat_list()
        )
        distorted_unit_depth_coords = linear_camera_cal.unit_depth_from_pixel(pixel)

        # Compute undistortion weight
        omega = self.distortion_coeffs[0]
        distorted_radius = distorted_unit_depth_coords.norm(epsilon)
        undistortion_weight = sm.tan(distorted_radius * omega) / (
            2 * distorted_radius * sm.tan(omega / 2.0)
        )

        # Apply weight and convert to camera ray
        unit_depth = undistortion_weight * distorted_unit_depth_coords
        camera_ray = geo.V3(unit_depth[0], unit_depth[1], 1)
        is_valid = sm.Max(sm.sign(sm.pi / 2 - sm.Abs(distorted_radius * omega)), 0)
        return camera_ray, is_valid
