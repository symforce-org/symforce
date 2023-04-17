# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import numpy as np

import symforce.symbolic as sf
from symforce.ops import StorageOps
from symforce.test_util import TestCase
from symforce.test_util.cam_cal_test_mixin import CamCalTestMixin
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


class EquirectangularTest(LieGroupOpsTestMixin, CamCalTestMixin, TestCase):
    """
    Test the EquirectangularCameraCal class.
    Note the mixin that tests all storage ops and camera projection/reprojection ops.
    """

    @classmethod
    def element(cls) -> sf.EquirectangularCameraCal:
        [f_x, f_y, c_x, c_y] = np.random.uniform(low=0.0, high=1000.0, size=(4,))
        return sf.EquirectangularCameraCal(focal_length=(f_x, f_y), principal_point=(c_x, c_y))

    def test_is_valid(self) -> None:
        """
        Tests if random points and pixels are correctly labeled as valid/invalid
        """
        for _ in range(10):
            cam_cal = self.element()
            point = sf.V3(np.random.uniform(low=-1.0, high=1.0, size=(3,)))

            with self.subTest(cam_cal=cam_cal, point=point):
                pixel, is_valid_forward_proj = cam_cal.pixel_from_camera_point(point)

                # Points at the origin should be invalid
                if point == point.zero():
                    self.assertStorageNear(is_valid_forward_proj, 0)
                else:
                    self.assertStorageNear(is_valid_forward_proj, 1)

                _, is_valid_back_proj = cam_cal.camera_ray_from_pixel(pixel)

                linear_camera_cal = sf.LinearCameraCal(
                    cam_cal.focal_length.to_flat_list(), cam_cal.principal_point.to_flat_list()
                )
                unit_depth = linear_camera_cal.unit_depth_from_pixel(pixel)
                if (
                    abs(StorageOps.evalf(unit_depth[0])) >= np.pi
                    or abs(StorageOps.evalf(unit_depth[1])) >= np.pi / 2.0
                ):
                    self.assertStorageNear(is_valid_back_proj, 0)
                else:
                    self.assertStorageNear(is_valid_back_proj, 1)

    def test_invalid_points(self) -> None:
        """
        Tests if specific invalid points are correctly labeled as invalid
        """
        invalid_points = [
            sf.V3.zero(),
        ]
        for point in invalid_points:
            for _ in range(10):
                cam_cal = self.element()
                _, is_valid_forward_proj = cam_cal.pixel_from_camera_point(point)
                self.assertTrue(is_valid_forward_proj == 0)

    def test_invalid_pixels(self) -> None:
        """
        Tests if specific invalid pixels are correctly labeled as invalid
        """
        f_x, f_y = (380, 380)
        c_x, c_y = (320, 240)
        cam_cal = sf.EquirectangularCameraCal(focal_length=(f_x, f_y), principal_point=(c_x, c_y))
        invalid_pixels = [
            sf.V2(f_x * (np.pi + 1e-6) + c_x, c_y),
            sf.V2(c_x, f_y * (np.pi / 2.0 + 1e-6) + c_y),
            sf.V2(f_x * (-np.pi - 1e-6) + c_x, c_y),
            sf.V2(c_x, f_y * (-np.pi / 2.0 - 1e-6) + c_y),
            sf.V2(1000, 1000),
            sf.V2(-1000, -1000),
        ]
        for pixel in invalid_pixels:
            _, is_valid_back_proj = cam_cal.camera_ray_from_pixel(pixel)
            self.assertTrue(StorageOps.evalf(is_valid_back_proj) == 0.0)


if __name__ == "__main__":
    TestCase.main()
