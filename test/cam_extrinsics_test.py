# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import numpy as np

import symforce.symbolic as sf
from symforce.test_util import TestCase
from symforce.test_util.cam_test_mixin import CamTestMixin


class CameraTest(CamTestMixin, TestCase):
    """
    Test the Camera class
    """

    @classmethod
    def element(cls) -> sf.Camera:
        return sf.Camera(
            calibration=sf.LinearCameraCal(focal_length=(440, 400), principal_point=(320, 240)),
            image_size=(640, 480),
        )

    def test_image_size(self) -> None:
        """
        Tests:
            Camera.in_view
            Camera.pixel_from_camera_point
            Camera.camera_ray_from_pixel
        """
        camera = self.element()
        assert camera.image_size is not None

        # Check that is_valid is set to 0 when point is outside FOV
        point_in_FOV = sf.V3(0, 0, 1)
        pixel_in_FOV, point_in_FOV_valid = camera.pixel_from_camera_point(point_in_FOV)
        self.assertTrue(point_in_FOV_valid == 1)
        self.assertTrue(camera.in_view(pixel_in_FOV, camera.image_size) == 1)

        point_outside_FOV = sf.V3(100, 0, 1)
        pixel_outside_FOV, point_outside_FOV_valid = camera.pixel_from_camera_point(
            point_outside_FOV
        )
        self.assertTrue(point_outside_FOV_valid == 0)
        self.assertTrue(camera.in_view(pixel_outside_FOV, camera.image_size) == 0)


class PosedCameraTest(CamTestMixin, TestCase):
    """
    Test the PosedCamera class
    """

    @classmethod
    def element(cls) -> sf.PosedCamera:
        return sf.PosedCamera(
            pose=sf.Pose3(R=sf.Rot3.from_yaw_pitch_roll(0.0, np.pi / 2.0, 0.0), t=sf.V3(0, 0, 100)),
            calibration=sf.LinearCameraCal(focal_length=(440, 400), principal_point=(320, 240)),
            image_size=(640, 480),
        )

    def test_posed_camera(self) -> None:
        """
        Tests:
            PosedCamera.pixel_from_global_point
            PosedCamera.global_point_from_pixel
        """
        posed_cam = self.element()

        # Transform some points to pixel coordinates and back
        for _ in range(100):
            global_point = (
                sf.V3(np.random.uniform(low=-1.0, high=1.0, size=(3,))) + posed_cam.pose.t
            )
            range_to_point = (global_point - posed_cam.pose.t).norm(epsilon=1e-9)
            pixel, is_valid = posed_cam.pixel_from_global_point(global_point)
            if is_valid == 1:
                # Make sure we reproject in the right direction
                global_point_reprojected, _ = posed_cam.global_point_from_pixel(
                    pixel, range_to_point=range_to_point
                )
                self.assertStorageNear(global_point, global_point_reprojected)

            # Transform pixel to global coordinates and back
            pixel = sf.V2(np.random.uniform(low=0, high=1000, size=(2,)))
            global_point, _ = posed_cam.global_point_from_pixel(pixel, range_to_point=1)
            pixel_reprojected, _ = posed_cam.pixel_from_global_point(global_point)
            self.assertStorageNear(pixel, pixel_reprojected)

    def test_warp_pixel(self) -> None:
        """
        Tests:
            PosedCamera.warp_pixel
        """
        # Create two cameras whose optical axes intersect
        posed_cam_1 = sf.PosedCamera(
            pose=sf.Pose3(
                R=sf.Rot3.from_yaw_pitch_roll(0.0, np.pi / 2.0, 0.0), t=sf.V3(0.0, 2.0, 0.0)
            ),
            calibration=sf.LinearCameraCal(focal_length=(440, 400), principal_point=(320, 240)),
        )
        posed_cam_2 = sf.PosedCamera(
            pose=sf.Pose3(
                R=sf.Rot3.from_yaw_pitch_roll(0.0, 0.0, -np.pi / 2.0), t=sf.V3(2.0, 0.0, 0.0)
            ),
            calibration=sf.LinearCameraCal(focal_length=(440, 400), principal_point=(320, 240)),
        )
        point_on_optical_axes = sf.V3(2.0, 2.0, 0.0)
        inverse_range = 0.5
        pixel_1, _ = posed_cam_1.pixel_from_global_point(point_on_optical_axes)
        pixel_2, is_valid_warp_into_2 = posed_cam_1.warp_pixel(
            pixel=pixel_1, inverse_range=inverse_range, target_cam=posed_cam_2
        )
        self.assertEqual(is_valid_warp_into_2, 1)
        self.assertStorageNear(pixel_1, pixel_2)

        # Try with a camera posed such that the point is invalid
        posed_cam_3 = sf.PosedCamera(
            pose=sf.Pose3(R=sf.Rot3(), t=sf.V3(0.0, 0.0, 1.0)),
            calibration=sf.LinearCameraCal(focal_length=(440, 400), principal_point=(320, 240)),
        )
        _, is_valid_warp_into_3 = posed_cam_1.warp_pixel(
            pixel=pixel_1, inverse_range=inverse_range, target_cam=posed_cam_3
        )
        self.assertEqual(is_valid_warp_into_3, 0)

        # Check that we don't get NaNs when using symbolic inverse_range and nonzero epsilon
        symbolic_inverse_range = sf.Symbol("inv_range")
        cam_1_ray = sf.V3(0.5, 1, 1)
        pixel_inf_1, _ = posed_cam_1.pixel_from_camera_point(cam_1_ray)
        pixel_inf_2, is_valid_inf = posed_cam_1.warp_pixel(
            pixel=pixel_inf_1,
            inverse_range=symbolic_inverse_range,
            target_cam=posed_cam_2,
            epsilon=self.EPSILON,
        )
        cam_2_ray = posed_cam_2.pose.R.inverse() * (posed_cam_1.pose.R * cam_1_ray)
        pixel_inf_2_rot_only, _ = posed_cam_2.pixel_from_camera_point(cam_2_ray)
        self.assertEqual(is_valid_inf.subs(symbolic_inverse_range, 0), 1)
        self.assertStorageNear(
            pixel_inf_2.subs(symbolic_inverse_range, 0), pixel_inf_2_rot_only, places=4
        )

        # Check that when inverse_range = 0 and epsilon = 0 we exactly recover rotation-only math
        pixel_inf_2_exact, is_valid_inf = posed_cam_1.warp_pixel(
            pixel=pixel_inf_1, inverse_range=0, target_cam=posed_cam_2
        )
        self.assertEqual(is_valid_inf, 1)
        self.assertStorageNear(pixel_inf_2_exact, pixel_inf_2_rot_only, places=9)


if __name__ == "__main__":
    TestCase.main()
