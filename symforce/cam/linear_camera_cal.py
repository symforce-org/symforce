from .camera_cal import CameraCal

from symforce import geo
from symforce import sympy as sm
from symforce import types as T


class LinearCameraCal(CameraCal):
    """
    Standard pinhole camera w/ four parameters [fx, fy, cx, cy].
    (fx, fy) representing focal length; (cx, cy) representing principal point.
    """

    NUM_DISTORTION_COEFFS = 0

    @staticmethod
    def project(point, epsilon=0):
        # type: (geo.Matrix31, T.Scalar) -> T.Tuple[geo.Matrix21, T.Scalar]
        """
        Linearly project the 3D point by dividing by the depth.

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
            z = sm.Max(sm.Abs(point[2]), epsilon)

        return geo.Vector2(x / z, y / z), sm.Max(sm.sign(point[2]), 0)

    def pixel_coords_from_unit_depth(self, unit_depth_coords):
        # type: (geo.Matrix31) -> geo.Matrix21
        """
        Convert point in unit-depth image plane to pixel coords by applying camera matrix.
        """
        return geo.Vector2(
            unit_depth_coords[0] * self.focal_length[0] + self.principal_point[0],
            unit_depth_coords[1] * self.focal_length[1] + self.principal_point[1],
        )

    def unit_depth_from_pixel_coords(self, pixel_coord):
        # type: (geo.Matrix21) -> geo.Matrix21
        """
        Convert point in pixel coordinates to unit-depth image plane by applying K_inv.
        """
        return geo.Vector2(
            (pixel_coord[0] - self.principal_point[0]) / self.focal_length[0],
            (pixel_coord[1] - self.principal_point[1]) / self.focal_length[1],
        )

    def pixel_coords_from_camera_point(self, point, epsilon=0):
        # type: (geo.Matrix31, T.Scalar) -> T.Tuple[geo.Matrix21, T.Scalar]
        unit_depth, is_valid = LinearCameraCal.project(point, epsilon=epsilon)
        return self.pixel_coords_from_unit_depth(unit_depth), is_valid

    def camera_ray_from_pixel_coords(self, pixel_coords, epsilon=0):
        # type: (geo.Matrix21, T.Scalar) -> T.Tuple[geo.Matrix31, T.Scalar]
        unit_depth = self.unit_depth_from_pixel_coords(pixel_coords)
        camera_ray = geo.Vector3(unit_depth[0], unit_depth[1], 1)
        is_valid = sm.S.One
        return camera_ray, is_valid
