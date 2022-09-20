# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import numpy as np

import symforce.symbolic as sf
from symforce.test_util import TestCase
from symforce.test_util.cam_cal_test_mixin import CamCalTestMixin
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


class CamLinearTest(LieGroupOpsTestMixin, CamCalTestMixin, TestCase):
    """
    Test the LinearCameraCal class.
    Note the mixin that tests all storage ops and camera projection/reprojection ops.
    """

    @classmethod
    def element(cls) -> sf.LinearCameraCal:
        [f_x, f_y, c_x, c_y] = np.random.uniform(low=0.0, high=1000.0, size=(4,))
        return sf.LinearCameraCal(focal_length=(f_x, f_y), principal_point=(c_x, c_y))

    def test_is_valid(self) -> None:
        """
        Tests if random points and pixels are correctly labeled as valid/invalid
        """
        for _ in range(10):
            cam_cal = self.element()
            point = sf.V3(np.random.uniform(low=-1.0, high=1.0, size=(3,)))
            pixel, is_valid_forward_proj = cam_cal.pixel_from_camera_point(point)

            # Points behind the camera should be invalid
            if point[2] > 0:
                self.assertTrue(is_valid_forward_proj == 1)
            else:
                self.assertTrue(is_valid_forward_proj == 0)

            _, is_valid_back_proj = cam_cal.camera_ray_from_pixel(pixel)

            # We should always be able to compute a valid ray from pixel coordinates for a linear camera
            self.assertTrue(is_valid_back_proj == 1)

    def test_invalid_points(self) -> None:
        """
        Tests if specific invalid points are correctly labeled as invalid
        """
        invalid_points = [
            sf.V3(0, 0, -1),
            sf.V3(0, 0, -1e-9),
            sf.V3(0, 0, -1000),
            sf.V3(1, 1, -1),
            sf.V3(-1, -1, -1),
            sf.V3(1000, 1000, -1000),
        ]
        for point in invalid_points:
            for _ in range(10):
                cam_cal = self.element()
                _, is_valid_forward_proj = cam_cal.pixel_from_camera_point(point)
                self.assertTrue(is_valid_forward_proj == 0)


if __name__ == "__main__":
    TestCase.main()
