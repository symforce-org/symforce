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


class PolynomialCameraCal(CameraCal):
    """
    Polynomial camera model in the style of OpenCV.
    Distortion is a multiplicitive factor applied to the image plane coordinates in the camera
    frame. Mapping between distorted image plane coordinates and image coordinates is done using
    a standard linear model.

    The distortion function is a 6th order even polynomial that is a function of the radius of the
    image plane coordinates.
    r = (p_img[0] ** 2 + p_img[1]**2) ** 0.5
    distorted_weight = 1 + c0 * r^2 + c1 * r^4 + c2 * r^6
    uv = p_img * distorted_weight

    """

    NUM_DISTORTION_COEFFS = 3
    DEFAULT_MAX_FOV = math.radians(120)

    def __init__(
        self,
        focal_length: T.Sequence[T.Scalar],
        principal_point: T.Sequence[T.Scalar],
        distortion_coeffs: T.Sequence[T.Scalar] = (0.0, 0.0, 0.0),
        critical_undistorted_radius: T.Scalar = None,
        max_fov: T.Scalar = DEFAULT_MAX_FOV,
    ) -> None:
        """
        Args:
            critical_undistorted_radius: The maximum radius allowable for distortion. This should
                                         be as large as largest expected undistorted radius. If the
                                         distortion coeffs are all numerical, this will be computed
                                         automatically as either the first local maximum of
                                         r(radius) in the interval (0, max_radius), or as max_radius
                                         if there is no local maximum in the interval.
            max_fov: Only used to compute critical_undistorted_radius when all camera parameters
                     are numerical. The maximum FOV (field of view) determines the maximum
                     image plane coordinates which is used to compute maximum radius.
        """
        super().__init__(focal_length, principal_point, distortion_coeffs)

        if critical_undistorted_radius is not None:
            self.critical_undistorted_radius = critical_undistorted_radius
        else:
            if any(
                isinstance(c, sf.Expr) and not isinstance(c, sf.Number) for c in distortion_coeffs
            ):
                raise ValueError(
                    "critical_undistorted_radius must be provided if the distortion_coeffs are not all numerical"
                )
            else:
                self.critical_undistorted_radius = self._compute_critical_undistorted_radius(
                    max_fov
                )

    @classmethod
    def from_distortion_coeffs(
        cls,
        focal_length: T.Sequence[T.Scalar],
        principal_point: T.Sequence[T.Scalar],
        distortion_coeffs: T.Sequence[T.Scalar] = tuple(),
        **kwargs: T.Scalar,
    ) -> PolynomialCameraCal:
        """
        Construct a Camera Cal of type cls from the focal_length, principal_point, and distortion_coeffs.

        kwargs are additional arguments which will be passed to the constructor.

        Symbolic arguments may only be passed if the kwarg critical_undistorted_radius is passed.
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
            ("critical_undistorted_radius", 1),
            ("distortion_coeffs", cls.NUM_DISTORTION_COEFFS),
        )

    def _distortion_weight(self, undistorted_radius: T.Scalar) -> T.Scalar:
        """
        Compute the distortion weight for the given undistorted radius. This weight is applied to
        the image plane coordinates.
        """
        total = 1.0
        radius_term = 1.0
        for coef in self.distortion_coeffs.to_flat_list():
            radius_term *= undistorted_radius ** 2
            total += radius_term * coef
        return total

    def pixel_from_camera_point(
        self, point: geo.Matrix31, epsilon: T.Scalar = sf.epsilon()
    ) -> T.Tuple[geo.Matrix21, T.Scalar]:
        p_img, project_is_valid = LinearCameraCal.project(point, epsilon)

        undistorted_radius = p_img.norm(epsilon)
        distortion_is_valid = sf.is_positive(self.critical_undistorted_radius - undistorted_radius)
        distorted_p_img = p_img * self._distortion_weight(undistorted_radius)

        linear_camera_cal = LinearCameraCal(
            self.focal_length.to_flat_list(), self.principal_point.to_flat_list()
        )
        uv = linear_camera_cal.pixel_from_unit_depth(distorted_p_img)

        is_valid = sf.logical_and(project_is_valid, distortion_is_valid, unsafe=True)

        return uv, is_valid

    def camera_ray_from_pixel(
        self, pixel: geo.Matrix21, epsilon: float = 0
    ) -> T.Tuple[geo.Matrix31, T.Scalar]:
        raise NotImplementedError(
            "Back projection involves computing the inverse of a polynomial function"
        )

    def _compute_critical_undistorted_radius(self, max_fov: float) -> float:
        """
        Compute the critical_undistorted_radius for the given (numerical) distortion_coeffs and
        max_fov. The max_fov is used as a bound for the max critical radius.
        """
        # The maximum radius in the image plane given the maximum expected FOV of the camera.
        # FOV is symmetric.
        max_radius = math.tan(max_fov / 2)
        return compute_odd_polynomial_critical_point(
            self.distortion_coeffs.to_flat_list(), max_radius
        )

    # -------------------------------------------------------------------------
    # Storage concept - see symforce.ops.storage_ops
    # -------------------------------------------------------------------------

    @classmethod
    def storage_dim(cls) -> int:
        # Contains the standard intrinsics, plus the critical_undistorted_radius
        return 4 + 1 + cls.NUM_DISTORTION_COEFFS

    def to_storage(self) -> T.List[T.Scalar]:
        return (
            self.focal_length.to_storage()
            + self.principal_point.to_storage()
            + [self.critical_undistorted_radius]
            + self.distortion_coeffs.to_storage()
        )

    @classmethod
    def from_storage(cls, vec: T.Sequence[T.Scalar]) -> PolynomialCameraCal:
        assert len(vec) == cls.storage_dim()
        return cls(
            focal_length=vec[0:2],
            principal_point=vec[2:4],
            critical_undistorted_radius=vec[4],
            distortion_coeffs=vec[5:],
        )

    @classmethod
    def symbolic(cls, name: str, **kwargs: T.Any) -> PolynomialCameraCal:
        with sf.scope(name):
            return cls(
                focal_length=sf.symbols("f_x f_y"),
                principal_point=sf.symbols("c_x c_y"),
                critical_undistorted_radius=sf.Symbol("radius_crit"),
                distortion_coeffs=geo.Matrix(cls.NUM_DISTORTION_COEFFS, 1)
                .symbolic("C", **kwargs)
                .to_flat_list(),
            )

    def __repr__(self) -> str:
        return "<{}\n  focal_length={},\n  principal_point={},\n  critical_undistorted_radius={},\n  distortion_coeffs={}>".format(
            self.__class__.__name__,
            self.focal_length.to_storage(),
            self.principal_point.to_storage(),
            self.critical_undistorted_radius,
            self.distortion_coeffs.to_storage(),
        )
