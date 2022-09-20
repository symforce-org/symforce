# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import numpy as np

import symforce.symbolic as sf
from symforce import typing as T
from symforce.ops import StorageOps
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
    def element(cls) -> T.Any:
        """
        Overriden by child to provide an example of a camera or camera calibration object.
        """
        raise NotImplementedError()

    def test_pixel_from_camera_point(self) -> None:
        """
        Tests:
            pixel_from_camera_point
        """
        if not self.element().has_camera_ray_from_pixel():
            # TODO(aaron, dominic, brad): Back projection is not implemented for SphericalCameraCal or
            # PolynomialCameraCal.
            self.skipTest(f"camera_ray_from_pixel not implemented for {type(self.element())}.")

        # Check that we can project a point in 3D into the image and back
        for _ in range(10):
            cam_cal = self.element()
            point = sf.V3(np.random.uniform(low=-1.0, high=1.0, size=(3,)))

            pixel, is_valid_forward_proj = cam_cal.pixel_from_camera_point(point)

            camera_ray, is_valid_back_proj = cam_cal.camera_ray_from_pixel(pixel)

            if abs(StorageOps.evalf(is_valid_forward_proj) - 1) < self.EPSILON:
                self.assertTrue(sf.Matrix.are_parallel(point, camera_ray, epsilon=self.EPSILON))
                self.assertStorageNear(is_valid_back_proj, 1)
            else:
                self.assertStorageNear(is_valid_forward_proj, 0)

    def test_camera_ray_from_pixel(self) -> None:
        """
        Tests:
            camera_ray_from_pixel
        """
        if not self.element().has_camera_ray_from_pixel():
            # TODO(aaron, dominic, brad): Back projection is not implemented for SphericalCameraCal or
            # PolynomialCameraCal.
            self.skipTest(f"camera_ray_from_pixel not implemented for {type(self.element())}.")

        # Check that we can project a point in the image into 3D and back
        for _ in range(10):
            cam_cal = self.element()

            # Try to generate pixels over a range that includes both valid and invalid pixel coordinates
            cx, cy = cam_cal.principal_point
            pixel = sf.V2(
                np.random.uniform(low=-0.5 * cx, high=2.5 * cx),
                np.random.uniform(low=-0.5 * cy, high=2.5 * cy),
            )

            camera_ray, is_valid_back_proj = cam_cal.camera_ray_from_pixel(pixel)

            (pixel_reprojected, is_valid_forward_proj) = cam_cal.pixel_from_camera_point(camera_ray)

            if abs(StorageOps.evalf(is_valid_back_proj) - 1) < self.EPSILON:
                self.assertStorageNear(pixel, pixel_reprojected)
                self.assertStorageNear(is_valid_forward_proj, 1)
            else:
                self.assertStorageNear(is_valid_back_proj, 0)
