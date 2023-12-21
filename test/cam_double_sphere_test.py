# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import numpy as np

import symforce.symbolic as sf
from symforce import typing as T
from symforce.test_util import TestCase
from symforce.test_util.cam_cal_test_mixin import CamCalTestMixin
from symforce.test_util.lie_group_ops_test_mixin import LieGroupOpsTestMixin


class DoubleSphereTest(LieGroupOpsTestMixin, CamCalTestMixin, TestCase):
    """
    Test the DoubleSphereCameraCal class.
    Note the mixin that tests all storage ops and camera projection/reprojection ops.
    """

    EPS = 1e-6

    @classmethod
    def element(cls) -> sf.DoubleSphereCameraCal:
        [f_x, f_y, c_x, c_y] = np.random.uniform(low=0.0, high=1000.0, size=(4,))
        xi = np.random.uniform(low=-1.0, high=10.0)
        alpha = np.random.uniform(high=0.9, low=-10.0)
        return sf.DoubleSphereCameraCal(
            focal_length=(f_x, f_y), principal_point=(c_x, c_y), xi=xi, alpha=alpha
        )

    @staticmethod
    def _make_cal(xi: float, alpha: float) -> sf.DoubleSphereCameraCal:
        focal_length = (800, 800)
        principal_point = (400, 400)

        return sf.DoubleSphereCameraCal(
            focal_length=focal_length,
            principal_point=principal_point,
            xi=xi,
            alpha=alpha,
        )

    def test_xi_and_alpha_properties(self) -> None:
        """
        Tests that the xi and alpha properties can be correctly read and written to.
        """
        xi, alpha = sf.symbols("xi alpha")
        cal = self._make_cal(xi=xi, alpha=alpha)

        with self.subTest(msg="Test getters"):
            self.assertEqual(cal.xi, xi)
            self.assertEqual(cal.alpha, alpha)

        with self.subTest(msg="Test setters"):
            cal.xi = alpha
            cal.alpha = xi
            self.assertEqual(cal.xi, alpha)
            self.assertEqual(cal.alpha, xi)

    def test_is_valid_forward(self) -> None:
        """
        Tests if strategically chosen points have valid projections
        """

        def point_from_angle(angle: float) -> sf.V3:
            """
            Generate a point a given angle away from the optical axis, with random distance from
            origin and random rotation about the optical axis
            """
            norm = np.random.uniform(0.1, 100)

            P = (
                sf.Rot3.from_angle_axis(np.random.uniform(0, 2 * np.pi), sf.V3(0, 0, 1))
                * sf.Rot3.from_angle_axis(angle=angle, axis=sf.V3(0, 1, 0))
                * sf.V3(0, 0, norm)
            )

            return P

        def check_forward_is_valid_on_boundary(xi: float, alpha: float, angle: float) -> None:
            """
            Check that is_valid is True on the valid side and False on the invalid side of a
            boundary
            """
            cal = self._make_cal(xi, alpha)

            with self.subTest(angle=angle, xi=xi, alpha=alpha):
                point = point_from_angle(angle - self.EPS)
                _, is_valid = cal.pixel_from_camera_point(point)
                self.assertEqual(T.cast(sf.Expr, is_valid).evalf(), 1.0)

            with self.subTest(angle=angle, xi=xi, alpha=alpha):
                point = point_from_angle(angle + self.EPS)
                _, is_valid = cal.pixel_from_camera_point(point)
                self.assertEqual(T.cast(sf.Expr, is_valid).evalf(), 0.0)

        # linear is_valid for trivial case
        check_forward_is_valid_on_boundary(0, 0, np.pi / 2)

        # linear is_valid for spheres overlapping, linear focal point inside second sphere
        check_forward_is_valid_on_boundary(0.3, 0, 1.8754889)

        # linear is_valid for spheres not overlapping, linear focal point near top of second sphere
        check_forward_is_valid_on_boundary(2, -10, 1.4145612)

        # linear is_valid for spheres overlapping, linear focal point outside second sphere
        check_forward_is_valid_on_boundary(0.3, 0.7, 2.2881935)

        # sphere is_valid for spheres far apart
        check_forward_is_valid_on_boundary(3, 0.7, 1.9106332)

    def test_is_valid_backward(self) -> None:
        """
        Test if strategically chosen pixels have valid backprojections
        """

        def pixel_from_radius(radius: float) -> sf.V2:
            """
            Generate a pixel a given radius away from the principal point, at a random angle
            """
            return (
                sf.Rot2.from_tangent([np.random.uniform(0, 2 * np.pi)]) * sf.V2(radius, 0)
            ) + sf.V2(400, 400)

        def check_backward_is_valid_on_boundary(xi: float, alpha: float, radius: float) -> None:
            """
            Check that is_valid is True on the valid side and False on the invalid side of a
            boundary
            """
            cal = self._make_cal(xi, alpha)

            with self.subTest(radius=radius, xi=xi, alpha=alpha):
                pixel = pixel_from_radius(radius - self.EPS)
                _, is_valid = cal.camera_ray_from_pixel(pixel)
                self.assertEqual(T.cast(sf.Expr, is_valid).evalf(), 1.0)

            with self.subTest(radius=radius, xi=xi, alpha=alpha):
                pixel = pixel_from_radius(radius + self.EPS)
                _, is_valid = cal.camera_ray_from_pixel(pixel)
                self.assertEqual(T.cast(sf.Expr, is_valid).evalf(), 0.0)

        # sphere is_valid for spheres far apart
        check_backward_is_valid_on_boundary(3, 0.7, 271.321813)

        # linear is_valid for spheres overlapping, linear focal point outside second sphere
        check_backward_is_valid_on_boundary(0.3, 0.7, 1264.911064)


if __name__ == "__main__":
    TestCase.main()
