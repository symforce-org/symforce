from .camera_cal import CameraCal
from .linear_camera_cal import LinearCameraCal

from symforce import geo
from symforce import sympy as sm
from symforce import types as T


class EquidistantEpipolarCameraCal(CameraCal):
    """
    Equidistant epipolar camera model with parameters [fx, fy, cx, cy].
    (fx, fy) representing focal length; (cx, cy) representing principal point.

    See here for more details:
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.473.5861&rep=rep1&type=pdf
    """

    NUM_DISTORTION_COEFFS = 0

    def pixel_from_camera_point(self, point, epsilon=0):
        # type: (geo.Matrix31, T.Scalar) -> T.Tuple[geo.Matrix21, T.Scalar]
        cam_yz_norm = sm.sqrt(point[1] ** 2 + point[2] ** 2 + epsilon)
        ud_x = sm.atan2(point[0], cam_yz_norm)
        ud_y = sm.atan2(point[1], point[2] + epsilon)

        linear_camera_cal = LinearCameraCal(self.focal_length, self.principal_point)
        pixel = linear_camera_cal.pixel_from_unit_depth(geo.V2(ud_x, ud_y))

        is_valid = sm.Max(sm.sign(point[2]), 0)
        return pixel, is_valid

    def camera_ray_from_pixel(self, pixel, epsilon=0):
        # type: (geo.Matrix21, T.Scalar) -> T.Tuple[geo.Matrix31, T.Scalar]
        linear_camera_cal = LinearCameraCal(self.focal_length, self.principal_point)
        unit_depth = linear_camera_cal.unit_depth_from_pixel(pixel)
        xyz = geo.V3(
            sm.sin(unit_depth[0]),
            sm.cos(unit_depth[0]) * sm.sin(unit_depth[1]),
            sm.cos(unit_depth[0]) * sm.cos(unit_depth[1]),
        )

        is_valid = sm.Mul(
            *[sm.Max(sm.sign(sm.pi / 2 - sm.Abs(p)), 0) for p in (unit_depth[0], unit_depth[1])]
        )
        return xyz, is_valid
