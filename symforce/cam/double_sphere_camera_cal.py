# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import symforce.internal.symbolic as sf
from symforce import geo
from symforce import typing as T
from symforce.cam.linear_camera_cal import LinearCameraCal

from .camera_cal import CameraCal


class DoubleSphereCameraCal(CameraCal):
    """
    Camera model where a point is consecutively projected onto two unit spheres
    with centers shifted by `xi`, then projected into the image plane using the
    pinhole model shifted by `alpha / (1 - alpha)`.

    There are important differences here from the derivation in the paper and in other open-source
    packages with double sphere models; see notebooks/double_sphere_derivation.ipynb for more
    information.

    The storage for this class is:
    [ fx fy cx cy xi alpha ]

    TODO(aaron): Create double_sphere_derivation.ipynb

    TODO(aaron): Probably want to check that values and derivatives are correct (or at least sane)
    on the valid side of the is_valid boundary

    Reference:
        https://vision.in.tum.de/research/vslam/double-sphere
    """

    NUM_DISTORTION_COEFFS = 2

    def __init__(
        self,
        focal_length: T.Sequence[T.Scalar],
        principal_point: T.Sequence[T.Scalar],
        xi: T.Scalar,
        alpha: T.Scalar,
    ) -> None:
        super().__init__(focal_length, principal_point, [xi, alpha])

    @classmethod
    def symbolic(cls, name: str, **kwargs: T.Any) -> DoubleSphereCameraCal:
        with sf.scope(name):
            return cls(
                focal_length=sf.symbols("f_x f_y"),
                principal_point=sf.symbols("c_x c_y"),
                xi=sf.Symbol("xi"),
                alpha=sf.Symbol("alpha"),
            )

    @property
    def xi(self) -> T.Scalar:
        return self.distortion_coeffs[0]

    @xi.setter
    def xi(self, value: T.Scalar) -> None:
        self.distortion_coeffs[0] = value

    @property
    def alpha(self) -> T.Scalar:
        return self.distortion_coeffs[1]

    @alpha.setter
    def alpha(self, value: T.Scalar) -> None:
        self.distortion_coeffs[1] = value

    @classmethod
    def storage_order(cls) -> T.Tuple[T.Tuple[str, int], ...]:
        return ("focal_length", 2), ("principal_point", 2), ("xi", 1), ("alpha", 1)

    def pixel_from_camera_point(
        self, point: geo.Matrix31, epsilon: T.Scalar = sf.epsilon()
    ) -> T.Tuple[geo.V2, T.Scalar]:
        # Pull out named scalar quantities
        x, y, z = point
        xi, alpha = self.distortion_coeffs

        # -1 if alpha < 0.5 else 1
        snz = sf.sign_no_zero(alpha - 0.5)

        # Protect for divide by zero
        # alpha_safe = sf.Max(epsilon, sf.Min(alpha, 1 - epsilon))
        alpha_safe = alpha - snz * epsilon

        # Follows equations (40) to (45)

        d1 = sf.sqrt(x ** 2 + y ** 2 + z ** 2 + epsilon ** 2)
        d2 = sf.sqrt(x ** 2 + y ** 2 + (xi * d1 + z) ** 2 + epsilon ** 2)

        z_effective = alpha_safe * d2 + (1 - alpha_safe) * (xi * d1 + z)

        # Image plane to pixel coordinate
        # NOTE(hayk, aaron): From the paper, the extra is_valid from the linear cam is redundant
        linear_camera_cal = LinearCameraCal(
            self.focal_length.to_flat_list(), self.principal_point.to_flat_list()
        )
        pixel, _ = linear_camera_cal.pixel_from_camera_point(
            geo.V3(x, y, z_effective), epsilon=epsilon
        )

        # w1 was simplified by hand
        w1 = ((snz + 1) / 2 - alpha_safe) / ((snz - 1) / 2 + alpha_safe)

        # NOTE(aaron): w2 here is NOT equal to the w2 in the paper - we're pretty confident this
        # one is correct though (for all domains, including the domain in the paper)
        w2_discriminant = w1 ** 2 * xi ** 2 - xi ** 2 + 1
        w2 = w1 ** 2 * xi - w1 * sf.sqrt(sf.Max(w2_discriminant, sf.sqrt(epsilon))) - xi

        need_linear_constraint = sf.is_nonnegative(w2_discriminant)

        linear_is_valid = sf.logical_or(
            sf.logical_not(need_linear_constraint, unsafe=True),
            sf.is_nonnegative(z - w2 * d1),
            unsafe=True,
        )

        # We also have the constraint that the unprojection from the second sphere to the first is
        # unique.  This is always satisfied for the domain in the paper, but we allow xi >= 1,
        # where this is not always satisfied
        need_sphere_constraint = sf.is_nonnegative(xi - 1)
        sphere_is_valid = sf.logical_or(
            sf.logical_not(need_sphere_constraint, unsafe=True),
            sf.is_nonnegative(z * xi + d1),
            unsafe=True,
        )

        return pixel, sf.logical_and(linear_is_valid, sphere_is_valid, unsafe=True)

    def camera_ray_from_pixel(
        self, pixel: geo.Matrix21, epsilon: T.Scalar = sf.epsilon()
    ) -> T.Tuple[geo.V3, T.Scalar]:
        # Pull out named scalar quantities
        xi, alpha = self.distortion_coeffs

        # Equations 47-49
        linear_camera_cal = LinearCameraCal(
            self.focal_length.to_flat_list(), self.principal_point.to_flat_list()
        )
        m_xy = linear_camera_cal.unit_depth_from_pixel(pixel)
        r2 = m_xy.squared_norm()

        # Compute m_z (eq 50)
        m_z_disciminant = 1 - (2 * alpha - 1) * r2
        linear_is_valid = sf.is_nonnegative(m_z_disciminant)

        # This denominator is not always positive so we push it away from 0, see:
        # https://www.wolframalpha.com/input/?i=Plot%5Balpha+*+Sqrt%5B1+-+%282+*+alpha+-+1%29+*+r%5E2%5D+%2B+1+-+alpha%2C+%7Balpha%2C+-2%2C+1%7D%2C+%7Br%2C+0%2C+10%7D%5D
        m_z_denominator = alpha * sf.sqrt(sf.Max(m_z_disciminant, epsilon)) + 1 - alpha
        m_z_denominator_safe = m_z_denominator + sf.sign_no_zero(m_z_denominator) * epsilon
        m_z = (1 - alpha ** 2 * r2) / m_z_denominator_safe

        # Compute the scalar multiplier on m (from eq 46)
        m_scale_denominator = m_z ** 2 + r2
        m_scale_denominator_safe = (
            m_scale_denominator + sf.sign_no_zero(m_scale_denominator) * epsilon
        )

        m_scale_discriminant = m_z ** 2 + (1 - xi ** 2) * r2

        # NOTE(aaron): This additional check is always satisfied when xi is strictly between -1 and
        # 1, but we allow xi > 1, where this becomes necessary.  The xi > 1 domain better fits some
        # cameras than restricting xi strictly between -1 and 1.
        sphere_is_valid = sf.is_nonnegative(m_scale_discriminant)

        m_scale = (
            m_z * xi + sf.sqrt(sf.Max(m_scale_discriminant, epsilon))
        ) / m_scale_denominator_safe

        point = m_scale * geo.V3(m_xy[0], m_xy[1], m_z) - geo.V3(0, 0, xi)

        return point, sf.logical_and(linear_is_valid, sphere_is_valid, unsafe=True)
