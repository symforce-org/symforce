from __future__ import annotations

import math
import numpy as np

from .camera_cal import CameraCal

from symforce.cam.linear_camera_cal import LinearCameraCal
from symforce import geo
from symforce import logger
from symforce import sympy as sm
from symforce import types as T


class SphericalCameraCal(CameraCal):
    """
    Kannala-Brandt camera model, where radial distortion is modeled relative to the 3D angle theta
    off the optical axis as opposed to radius within the image plane (i.e. ATANCamera)

    I.e. the radius in the image plane as a function of the angle theta from the camera z-axis is
    assumed to be given by:

        r(theta) = theta + d[0] * theta^3 + d[1] * theta^5 + d[2] * theta^7 + d[3] * theta^9

    With no tangential coefficients, this model is over-parameterized in that we may scale all the
    distortion coefficients by a constant, and the focal length by the inverse of that constant. To
    fix this issue, we peg the first coefficient at 1. So while the distortion dimension is '4',
    the actual total number of coeffs is 5.

    Paper:
    A generic camera model and calibration method for conventional, wide-angle, and fish-eye lenses
    Kannala, Juho; Brandt, Sami S.
    PAMI 2006

    This is the simpler "P9" model without any non-radially-symmetric distortion params.
    """

    NUM_DISTORTION_COEFFS = 4

    def __init__(
        self,
        focal_length: T.Sequence[T.Scalar],
        principal_point: T.Sequence[T.Scalar],
        distortion_coeffs: T.Sequence[T.Scalar] = (0.0, 0.0, 0.0, 0.0),
        critical_theta: T.Scalar = None,
        max_theta: T.Scalar = math.pi,
    ) -> None:
        """
        Args:
            critical_theta: The maximum angle from the optical axis for which the projection is
                            valid.  In general, this should be at least as large as the FOV of the
                            camera.  If the distortion coeffs are all numerical, this will be
                            computed automatically as either the first local maximum of r(theta) in
                            the interval (0, max_theta), or as max_theta if there is no local
                            maximum in the interval.
            max_theta: Used only to compute critical_theta when the distortion coefficients are all
                       numerical, see the description for critical_theta.  The default value of 180
                       degrees should generally be fine regardless of the actual field of view
        """
        super(SphericalCameraCal, self).__init__(focal_length, principal_point, distortion_coeffs)

        if critical_theta is not None:
            self.critical_theta = critical_theta
        else:
            if any(
                [isinstance(c, sm.Expr) and not isinstance(c, sm.Number) for c in distortion_coeffs]
            ):
                raise ValueError(
                    "critical_theta must be provided if the distortion_coeffs are not all numerical"
                )
            else:
                self.critical_theta = self._compute_critical_theta(max_theta)

    # -------------------------------------------------------------------------
    # Storage concept - see symforce.ops.storage_ops
    # -------------------------------------------------------------------------

    @classmethod
    def storage_dim(cls) -> int:
        # Contains the standard intrinsics, plus the critical_theta
        return 4 + 1 + cls.NUM_DISTORTION_COEFFS

    def to_storage(self) -> T.List[T.Scalar]:
        return (
            self.focal_length.to_storage()
            + self.principal_point.to_storage()
            + [self.critical_theta]
            + self.distortion_coeffs.to_storage()
        )

    @classmethod
    def from_storage(cls, vec: T.Sequence[T.Scalar]) -> SphericalCameraCal:
        assert len(vec) == cls.storage_dim()
        return cls(
            focal_length=vec[0:2],
            principal_point=vec[2:4],
            critical_theta=vec[4],
            distortion_coeffs=vec[5:],
        )

    @classmethod
    def symbolic(cls, name: str, **kwargs: T.Any) -> SphericalCameraCal:
        with sm.scope(name):
            if cls.NUM_DISTORTION_COEFFS > 0:
                return cls(
                    focal_length=sm.symbols("f_x f_y"),
                    principal_point=sm.symbols("c_x c_y"),
                    critical_theta=sm.Symbol("theta_crit"),
                    distortion_coeffs=geo.Matrix(cls.NUM_DISTORTION_COEFFS, 1)
                    .symbolic("C", **kwargs)
                    .to_flat_list(),
                )
            else:
                return cls(
                    focal_length=sm.symbols("f_x f_y"),
                    principal_point=sm.symbols("c_x c_y"),
                    critical_theta=sm.Symbol("theta_c"),
                )

    def __repr__(self) -> str:
        return "<{}\n  focal_length={},\n  principal_point={},\n  critical_theta={},\n  distortion_coeffs={}>".format(
            self.__class__.__name__,
            self.focal_length.to_storage(),
            self.principal_point.to_storage(),
            self.critical_theta,
            self.distortion_coeffs.to_storage(),
        )

    def _compute_critical_theta(self, max_theta: float) -> float:
        """
        Compute the critical_theta for the given (numerical) distortion_coeffs and max_theta
        """
        # Build the coefficients for np.polynomial.  Even coefficients are 0, and the coefficient
        # on theta is 1
        full_poly_coeffs = [0.0, 1.0]
        for k in self.distortion_coeffs.to_flat_list():
            full_poly_coeffs.extend([0.0, float(k)])
        critical_points = np.polynomial.Polynomial(np.array(full_poly_coeffs)).deriv().roots()

        # NOTE(aaron): This is a tolerance on the result of `np.roots` so it doesn't really have
        # anything to do with epsilon or anything.  Could be worth investigating the actual error
        # properties on that, but the docs don't say
        ROOTS_REAL_TOLERANCE = 1e-10

        real_critical_points = critical_points[
            abs(critical_points.imag) < ROOTS_REAL_TOLERANCE
        ].real

        real_critical_points_in_interval = np.sort(
            real_critical_points[
                np.logical_and(real_critical_points > 0, real_critical_points < max_theta)
            ]
        )

        if real_critical_points_in_interval.size == 0:
            return max_theta
        else:
            return real_critical_points_in_interval[0]

    def _radial_distortion(self, theta: sm.Symbol) -> sm.Symbol:
        """
        Compute the radius in the unit-depth plane from the angle theta with the camera z-axis
        """
        theta_term = theta
        acc = theta

        for coef in self.distortion_coeffs.to_flat_list():
            theta_term *= theta ** 2
            acc += coef * theta_term

        return acc

    def pixel_from_camera_point(
        self, point: geo.Matrix31, epsilon: T.Scalar = 0,
    ) -> T.Tuple[geo.Matrix21, T.Scalar]:

        # compute theta
        xy_norm = point[:2, :].norm(epsilon)
        theta = sm.atan2(xy_norm, point[2])
        is_valid = sm.Max(sm.sign(self.critical_theta - theta), 0)

        # clamp theta to critical_theta
        theta = sm.Min(theta, self.critical_theta - epsilon)

        # compute image plane coordinate
        r = self._radial_distortion(theta)
        unit_xy = point[:2, :].normalized(epsilon)
        point_img = r * unit_xy

        # image plane to pixel coordinate
        linear_camera_cal = LinearCameraCal(
            self.focal_length.to_flat_list(), self.principal_point.to_flat_list()
        )
        point_pix = linear_camera_cal.pixel_from_unit_depth(point_img)

        return point_pix, is_valid

    def camera_ray_from_pixel(
        self, pixel: geo.Matrix21, epsilon: float = 0
    ) -> T.Tuple[geo.Matrix31, T.Scalar]:
        raise NotImplementedError(
            "Back projection involves computing the inverse of a polynomial function"
        )
