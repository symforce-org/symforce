# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from symforce import typing as T


def compute_odd_polynomial_critical_point(
    coefficients: T.Iterable[T.Scalar], max_x: float
) -> float:
    """
    Compute the critical points of the odd polynomial given by:
    f(x) = x + c0 * x^3 + c1 * x^5 + ...
    This will return the first real-valued critical point in the range [0, max]. If no real-valued
    critical points are found in this range, return max.

    Args:
            coefficients: the coefficients of the polynomial
            max_x: the maximum value to be returned if no real-valued critical points are found
                   in [0, max_x]
    """
    # Build the coefficients for np.polynomial.
    full_poly_coeffs = [0.0, 1.0]
    for k in coefficients:
        full_poly_coeffs.extend([0.0, float(k)])
    critical_points = np.polynomial.Polynomial(np.array(full_poly_coeffs)).deriv().roots()

    # NOTE(aaron): This is a tolerance on the result of `np.roots` so it doesn't really have
    # anything to do with epsilon or anything.  Could be worth investigating the actual error
    # properties on that, but the docs don't say
    ROOTS_REAL_TOLERANCE = 1e-10

    real_critical_points = critical_points[abs(critical_points.imag) < ROOTS_REAL_TOLERANCE].real

    real_critical_points_in_interval = np.sort(
        real_critical_points[np.logical_and(real_critical_points > 0, real_critical_points < max_x)]
    )

    if real_critical_points_in_interval.size == 0:
        return max_x
    else:
        return real_critical_points_in_interval[0]
