# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import numpy as np

import symforce.symbolic as sf
from symforce.test_util import TestCase
from symforce.test_util.cam_cal_test_mixin import CamCalTestMixin
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


class CamPolynomialTest(LieGroupOpsTestMixin, CamCalTestMixin, TestCase):
    """
    Test the Polynomial camera model class.
    """

    @classmethod
    def element(cls) -> sf.PolynomialCameraCal:
        [f_x, f_y, c_x, c_y] = np.random.uniform(low=0.0, high=1000.0, size=(4,))
        radial_coeffs = np.random.normal(scale=[0.1, 0.01, 0.001]).tolist()
        return sf.PolynomialCameraCal(
            focal_length=(f_x, f_y), principal_point=(c_x, c_y), distortion_coeffs=radial_coeffs
        )

    def test_max_critical_theta(self) -> None:
        # Camera with no forced critical value. Expected to be derived from the default max angle.
        default_max_radius = np.tan(sf.PolynomialCameraCal.DEFAULT_MAX_FOV / 2)
        cal = sf.PolynomialCameraCal((300.0, 300.0), (500.0, 500.0), (0.0, 0.0, 0.0))
        self.assertEqual(cal.critical_undistorted_radius, default_max_radius)

        # Camera with a critical point. Computed by hand.
        cal = sf.PolynomialCameraCal((300.0, 300.0), (500.0, 500.0), (-4 / 3, 0.0, 0.0))
        self.assertStorageNear(cal.critical_undistorted_radius, 0.5)

    def test_projection_valid(self) -> None:
        # Critical point is 0.5 as seen above.
        cal = sf.PolynomialCameraCal((1.0, 1.0), (0.0, 0.0), (-4 / 3, 0.0, 0.0))

        # Compute the corresponding maximum angle expected to be valid.
        max_angle = np.arctan(0.5)
        valid_angle = max_angle * (1 - 0.001)
        invalid_angle = max_angle * (1 + 0.001)
        unit_z = sf.V3(0, 0, 1)

        for theta in np.linspace(0, 2 * np.pi, 10):
            rotation = sf.Rot3.from_yaw_pitch_roll(np.rad2deg(theta), 0, valid_angle)
            p_cam = rotation * unit_z
            _, is_valid = cal.pixel_from_camera_point(p_cam)
            self.assertStorageNear(sf.S(is_valid).evalf(), 1.0)

        for theta in np.linspace(0, 2 * np.pi, 10):
            rotation = sf.Rot3.from_yaw_pitch_roll(np.rad2deg(theta), 0, invalid_angle)
            p_cam = rotation * unit_z
            _, is_valid = cal.pixel_from_camera_point(p_cam)
            self.assertStorageNear(sf.S(is_valid).evalf(), 0.0)


if __name__ == "__main__":
    TestCase.main()
