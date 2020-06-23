import numpy as np

from symforce import cam
from symforce import geo
from symforce.test_util import TestCase
from symforce.test_util.storage_ops_test_mixin import StorageOpsTestMixin
from symforce.test_util.cam_test_mixin import CamTestMixin


class CamLinearTest(StorageOpsTestMixin, CamTestMixin, TestCase):
    """
    Test the LinearCameraCal class.
    Note the mixin that tests all storage ops and camera projection/reprojection ops.
    """

    @classmethod
    def element(cls):
        # type: () -> cam.LinearCameraCal
        [f_x, f_y, c_x, c_y] = np.random.uniform(low=0.0, high=1000.0, size=(4,))
        return cam.LinearCameraCal(focal_length=(f_x, f_y), principal_point=(c_x, c_y))

    def test_is_valid(self):
        # type: () -> None
        for _ in range(10):
            cam = self.element()
            point = geo.V3(np.random.uniform(low=-1.0, high=1.0, size=(3,)))
            pixel_coords, is_valid_forward_proj = cam.pixel_coords_from_camera_point(point)

            # Points behind the camera should be invalid
            if point[2] > 0:
                self.assertTrue(is_valid_forward_proj == 1)
            else:
                self.assertTrue(is_valid_forward_proj == 0)

            _, is_valid_back_proj = cam.camera_ray_from_pixel_coords(pixel_coords)

            # We should always be able to compute a valid ray from pixel coordinates for a linear camera
            self.assertTrue(is_valid_back_proj == 1)


if __name__ == "__main__":
    TestCase.main()
