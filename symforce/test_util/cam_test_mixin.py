import numpy as np

from symforce import sympy as sm
from symforce import geo
from symforce import types as T
from symforce.test_util import TestCase

if T.TYPE_CHECKING:
    _Base = TestCase
else:
    _Base = object


class CamTestMixin(_Base):
    """
    Test helper for camera objects. Inherit a test case from this.
    """

    # Small number to avoid singularities
    EPSILON = 1e-8

    @classmethod
    def element(cls):
        # type: () -> T.Any
        """
        Overriden by child to provide an example of a camera or camera calibration object.
        """
        raise NotImplementedError()

    def test_pixel_coords_from_camera_point(self):
        # type: () -> None
        """
        Tests:
            pixel_coords_from_camera_point
        """
        # Check that we can project a point in 3D into the image and back
        for _ in range(10):
            cam = self.element()
            point = geo.V3(np.random.uniform(low=-1.0, high=1.0, size=(3,)))

            pixel_coords, is_valid_forward_proj = cam.pixel_coords_from_camera_point(point)
            camera_ray, is_valid_back_proj = cam.camera_ray_from_pixel_coords(pixel_coords)

            if is_valid_forward_proj == 1:
                self.assertTrue(geo.Matrix.are_parallel(point, camera_ray, epsilon=self.EPSILON))
                self.assertTrue(is_valid_back_proj == 1)

    def test_camera_ray_from_pixel_coords(self):
        # type: () -> None
        """
        Tests:
            camera_ray_from_pixel_coords
        """
        # Check that we can project a point in the image into 3D and back
        for _ in range(10):
            cam = self.element()

            # Try to generate pixels over a range that includes both valid and invalid pixel coordinates
            cx, cy = cam.principal_point
            pixel_coords = geo.V2(
                np.random.uniform(low=-0.5 * cx, high=2.5 * cx),
                np.random.uniform(low=-0.5 * cy, high=2.5 * cy),
            )

            camera_ray, is_valid_back_proj = cam.camera_ray_from_pixel_coords(pixel_coords)
            pixel_coords_reprojected, is_valid_forward_proj = cam.pixel_coords_from_camera_point(
                camera_ray
            )

            if is_valid_back_proj == 1:
                self.assertNear(pixel_coords, pixel_coords_reprojected)
                self.assertTrue(is_valid_forward_proj == 1)
