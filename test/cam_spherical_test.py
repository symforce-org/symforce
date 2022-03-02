# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import numpy as np

from symforce import cam
from symforce import geo
from symforce import sympy as sm
from symforce.ops import StorageOps
from symforce.test_util import TestCase
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


class CamSphericalTest(LieGroupOpsTestMixin, TestCase):
    """
    Test the SphericalCameraCal class.

    TODO(aaron): Back projection is not implemented so cannot test with CamTestMixin.
    """

    @classmethod
    def element(cls) -> cam.SphericalCameraCal:
        [f_x, f_y, c_x, c_y] = np.random.uniform(low=0.0, high=1000.0, size=(4,))
        radial_coeffs = np.random.normal(scale=[0.1, 0.01, 0.001, 0.0001]).tolist()
        return cam.SphericalCameraCal(
            focal_length=(f_x, f_y), principal_point=(c_x, c_y), distortion_coeffs=radial_coeffs
        )

    def test_max_critical_theta(self) -> None:
        # spherical camera with no forced critical value
        cal = cam.SphericalCameraCal((300.0, 300.0), (500.0, 500.0), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(cal.critical_theta, np.pi)

        # spherical camera with a critical point
        cal = cam.SphericalCameraCal((300.0, 300.0), (500.0, 500.0), (-0.1, 0.0, 0.0, 0.0))
        self.assertStorageNear(cal.critical_theta, 1.8257419)

        # spherical camera with a critical point, but also imaginary ones with smaller real parts
        cal = cam.SphericalCameraCal(
            (300.0, 300.0),
            (500.0, 500.0),
            (0.0388132374, -0.028164965, 0.0077940788, -0.0015448705),
        )
        self.assertStorageNear(cal.critical_theta, 1.8565507)

        # spherical camera with two critical points, should pick the first
        cal = cam.SphericalCameraCal((300.0, 300.0), (500.0, 500.0), (-0.3, 0.035, 0.0, 0.0))
        self.assertStorageNear(cal.critical_theta, 1.2742925)

    def test_projection_valid(self) -> None:
        # r(theta) = theta - (1/3) * theta^3
        cal = cam.SphericalCameraCal((1.0, 1.0), (0.0, 0.0), (-1.0 / 3, 0.0, 0.0, 0.0))

        # by construction, projection should be valid for theta <= 1, invalid for theta > 1
        valid_point = geo.V3(1.0, 0.0, 1.0)
        _, is_valid = cal.pixel_from_camera_point(valid_point)
        self.assertStorageNear(sm.S(is_valid).evalf(), 1.0)

        invalid_point = geo.V3(1.0, 0.0, 0.0)
        _, is_valid = cal.pixel_from_camera_point(invalid_point)
        self.assertStorageNear(sm.S(is_valid).evalf(), 0.0)


if __name__ == "__main__":
    TestCase.main()
