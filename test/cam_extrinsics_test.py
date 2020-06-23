import numpy as np

from symforce import geo
from symforce import cam
from symforce.test_util import TestCase
from symforce.test_util.cam_test_mixin import CamTestMixin


class CameraTest(CamTestMixin, TestCase):
    """
    Test the Camera class
    """

    @classmethod
    def element(cls):
        # type: () -> cam.Camera
        return cam.Camera(
            calibration=cam.LinearCameraCal(focal_length=(440, 400), principal_point=(320, 240)),
            image_size=(640, 480),
        )

    def test_image_size(self):
        # type: () -> None
        cam = self.element()

        # Check that is_valid is set to 0 when point is outside FOV
        point_in_FOV = geo.V3(0, 0, 1)
        _, point_in_FOV_valid = cam.pixel_coords_from_camera_point(point_in_FOV)
        self.assertTrue(point_in_FOV_valid == 1)

        point_outside_FOV = geo.V3(100, 0, 1)
        _, point_outside_FOV_valid = cam.pixel_coords_from_camera_point(point_outside_FOV)
        self.assertTrue(point_outside_FOV_valid == 0)


class PosedCameraTest(CamTestMixin, TestCase):
    """
    Test the PosedCamera class
    """

    @classmethod
    def element(cls):
        # type: () -> cam.PosedCamera
        return cam.PosedCamera(
            pose=geo.Pose3(R=geo.Rot3.from_euler_ypr(0.0, np.pi / 2.0, 0.0), t=geo.V3(0, 0, 100)),
            calibration=cam.LinearCameraCal(focal_length=(440, 400), principal_point=(320, 240)),
            image_size=(640, 480),
        )

    def test_posed_camera(self):
        # type: () -> None
        cam = self.element()

        # Transform some points to pixel coordinates and back
        for _ in range(100):
            global_point = geo.V3(np.random.uniform(low=-1.0, high=1.0, size=(3,))) + cam.pose.t
            range_to_point = (global_point - cam.pose.t).norm(epsilon=1e-9)
            pixel_coords, is_valid = cam.pixel_coords_from_global_point(global_point)
            if is_valid == 1:
                # Make sure we reproject in the right direction
                global_point_reprojected, _ = cam.global_point_from_pixel_coords(
                    pixel_coords, range_to_point=range_to_point
                )
                self.assertNear(global_point, global_point_reprojected)

            # Transform pixel to global coordinates and back
            pixel_coords = geo.V2(np.random.uniform(low=0, high=1000, size=(2,)))
            global_point, _ = cam.global_point_from_pixel_coords(pixel_coords, range_to_point=1)
            pixel_coords_reprojected, _ = cam.pixel_coords_from_global_point(global_point)
            self.assertNear(pixel_coords, pixel_coords_reprojected)


if __name__ == "__main__":
    TestCase.main()
