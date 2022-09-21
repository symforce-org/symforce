# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import math

import symforce.internal.symbolic as sf
from symforce import geo
from symforce import typing as T
from symforce.cam.camera_util import compute_odd_polynomial_critical_point
from symforce.cam.linear_camera_cal import LinearCameraCal

from .camera_cal import CameraCal


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

    Additionally, the storage for this class includes the critical theta, the maximum angle from the
    optical axis where projection is invertible; although the critical theta is a function of the
    other parameters, this function requires polynomial root finding, so it should be computed
    externally at runtime and set to the computed value.

    Paper:
    A generic camera model and calibration method for conventional, wide-angle, and fish-eye lenses
    Kannala, Juho; Brandt, Sami S.
    PAMI 2006

    This is the simpler "P9" model without any non-radially-symmetric distortion params.

    The storage for this class is:
    [ fx fy cx cy critical_theta d0 d1 d2 d3 ]
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
        super().__init__(focal_length, principal_point, distortion_coeffs)

        if critical_theta is not None:
            self.critical_theta = critical_theta
        else:
            if any(
                isinstance(c, sf.Expr) and not isinstance(c, sf.Number) for c in distortion_coeffs
            ):
                raise ValueError(
                    "critical_theta must be provided if the distortion_coeffs are not all numerical"
                )
            else:
                self.critical_theta = self._compute_critical_theta(max_theta)

    @classmethod
    def from_distortion_coeffs(
        cls,
        focal_length: T.Sequence[T.Scalar],
        principal_point: T.Sequence[T.Scalar],
        distortion_coeffs: T.Sequence[T.Scalar] = tuple(),
        **kwargs: T.Scalar,
    ) -> SphericalCameraCal:
        """
        Construct a Camera Cal of type cls from the focal_length, principal_point, and distortion_coeffs.

        kwargs are additional arguments which will be passed to the constructor.

        Symbolic arguments may only be passed if the kwarg critical_theta is passed.
        """
        return cls(
            focal_length=focal_length,
            principal_point=principal_point,
            distortion_coeffs=distortion_coeffs,
            **kwargs,
        )

    @classmethod
    def storage_order(cls) -> T.Tuple[T.Tuple[str, int], ...]:
        return (
            ("focal_length", 2),
            ("principal_point", 2),
            ("critical_theta", 1),
            ("distortion_coeffs", cls.NUM_DISTORTION_COEFFS),
        )

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
        with sf.scope(name):
            return cls(
                focal_length=sf.symbols("f_x f_y"),
                principal_point=sf.symbols("c_x c_y"),
                critical_theta=sf.Symbol("theta_crit"),
                distortion_coeffs=geo.Matrix(cls.NUM_DISTORTION_COEFFS, 1)
                .symbolic("C", **kwargs)
                .to_flat_list(),
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
        return compute_odd_polynomial_critical_point(
            self.distortion_coeffs.to_flat_list(), max_theta
        )

    def _radial_distortion(self, theta: sf.Symbol) -> sf.Symbol:
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
        self, point: geo.Matrix31, epsilon: T.Scalar = sf.epsilon()
    ) -> T.Tuple[geo.Matrix21, T.Scalar]:

        # compute theta
        xy_norm = point[:2, :].norm(epsilon)
        theta = sf.atan2(xy_norm, point[2], epsilon=0)
        is_valid = sf.Max(sf.sign(self.critical_theta - theta), 0)

        # clamp theta to critical_theta
        theta = sf.Min(theta, self.critical_theta - epsilon)

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
