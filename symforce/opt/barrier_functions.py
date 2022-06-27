# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce.symbolic as sf


def max_power_barrier(
    x: sf.Scalar,
    x_nominal: sf.Scalar,
    error_nominal: sf.Scalar,
    dist_zero_to_nominal: sf.Scalar,
    power: sf.Scalar,
) -> sf.Scalar:
    """
    A one-sided, non-symmetric scalar barrier function. The barrier passes through the points
    (x_nominal, error_nominal) and (x_nominal - dist_zero_to_nominal, 0) with a curve of the
    form x^power. The parameterization of the barrier by these variables is convenient because it
    allows setting a constant penalty for a nominal point, then adjusting the `width` and
    `steepness` of the curve independently. The barrier with power = 1 will look like:

                                 |              **
                                 |             ** - (x_nominal, error_nominal) is a fixed point
                                 |            **
                                 |           **   <- x^power is the shape of the curve
                                 |          **
                                 |         **
             ----------*********************---------
                                 |         |<-->| dist_zero_to_nominal is the distance from
                                                  x_nominal to the point at which the error is zero

    Note that when applying the barrier function to a residual used in a least-squares problem, a
    power = 1 will lead to a quadratic cost in the optimization problem because the cost
    equals 1/2 * residual^2. For example:

    Cost (1/2 * residual^2) when the residual is a max_power_barrier with power = 1 (shown above):

                                |                *
                                |               ** - (x_nominal, error_nominal^2)
                                |               *
                                |              ** <- x^(2*power) is the shape of the cost curve
                                |            ***
                                |          ***
             ---------*********************---------
                                |         |<-->| dist_zero_to_nominal

    Args:
        x: The point at which we want to evaluate the barrier function.
        x_nominal: x-value of the point at which the error is equal to error_nominal.
        error_nominal: Error returned when x equals x_nominal.
        dist_zero_to_nominal: The distance from x_nominal to the region of zero error. Must be
            a positive number.
        power: The power used to describe the curve of the error tails.
    """
    x_zero_error = x_nominal - dist_zero_to_nominal
    return error_nominal * sf.Pow(sf.Max(0, x - x_zero_error) / dist_zero_to_nominal, power)


def max_linear_barrier(
    x: sf.Scalar, x_nominal: sf.Scalar, error_nominal: sf.Scalar, dist_zero_to_nominal: sf.Scalar
) -> sf.Scalar:
    """
    Applies "max_power_barrier" with power = 1.  When applied to a residual of a least-squares
    problem, this produces a quadratic cost in the optimization problem because
    cost = 1/2 * residual^2. See "max_power_barrier" for more details.
    """
    return max_power_barrier(
        x=x,
        x_nominal=x_nominal,
        error_nominal=error_nominal,
        dist_zero_to_nominal=dist_zero_to_nominal,
        power=1,
    )


def min_power_barrier(
    x: sf.Scalar,
    x_nominal: sf.Scalar,
    error_nominal: sf.Scalar,
    dist_zero_to_nominal: sf.Scalar,
    power: sf.Scalar,
) -> sf.Scalar:
    """
    A one-sided, non-symmetric scalar barrier function. The barrier passes through the points
    (x_nominal, error_nominal) and (x_nominal + dist_zero_to_nominal, 0) with a curve of the
    form x^power. The barrier with power = 1 will look like:

                                        **               |
            (x_nominal, error_nominal) - **              |
                                          **             |
    x^power is the shape of the curve ->   **            |
                                            **           |
                                             **          |
                                     ---------**********************---------
                    dist_zero_to_nominal  |<->|          |

    Args:
        x: The point at which we want to evaluate the barrier function.
        x_nominal: x-value of the point at which the error is equal to error_nominal.
        error_nominal: Error returned when x equals x_nominal.
        dist_zero_to_nominal: The distance from x_nominal to the region of zero error. Must be
            a positive number.
        power: The power used to describe the curve of the error tails. Note that
            when applying the barrier function to a residual used in a least-squares problem,
            a power = 1 will lead to a quadratic cost in the optimization problem.
    """
    # Flip x and x_nominal about the y-axis and reuse the max_power_barrier implementation
    return max_power_barrier(
        x=-x,
        x_nominal=-x_nominal,
        error_nominal=error_nominal,
        dist_zero_to_nominal=dist_zero_to_nominal,
        power=power,
    )


def min_linear_barrier(
    x: sf.Scalar, x_nominal: sf.Scalar, error_nominal: sf.Scalar, dist_zero_to_nominal: sf.Scalar
) -> sf.Scalar:
    """
    Applies "min_power_barrier" with power = 1.  When applied to a residual of a least-squares
    problem, this produces a quadratic cost in the optimization problem because
    cost = 1/2 * residual^2. See "min_power_barrier" for more details.
    """
    return min_power_barrier(
        x=x,
        x_nominal=x_nominal,
        error_nominal=error_nominal,
        dist_zero_to_nominal=dist_zero_to_nominal,
        power=1,
    )


def symmetric_power_barrier(
    x: sf.Scalar,
    x_nominal: sf.Scalar,
    error_nominal: sf.Scalar,
    dist_zero_to_nominal: sf.Scalar,
    power: sf.Scalar,
) -> sf.Scalar:
    """
    A symmetric barrier cenetered around x = 0, meaning the error at -x is equal to the error at x.
    The barrier passes through the points (x_nominal, error_nominal) and
    (x_nominal - dist_zero_to_nominal, 0) with a curve of the form x^power. For example, the
    barrier with power = 1 will look like:

                 **              |              **
                  **             |             ** - (x_nominal, error_nominal) is a fixed point
                   **            |            **
                    **           |           **   <- x^power is the shape of the curve
                     **          |          **
                      **         |         **
             ----------*********************---------
                                 |         |<-->| dist_zero_to_nominal is the distance from
                                                  x_nominal to the point at which the error is zero

    Note that when applying the barrier function to a residual used in a least-squares problem, a
    power = 1 will lead to a quadratic cost in the optimization problem because the cost
    equals 1/2 * residual^2. For example:

    Cost (1/2 * residual^2) when the residual is a symmetric barrier with power = 1 (shown above):

               *                |                *
               **               |               ** - (x_nominal, 1/2 * error_nominal^2)
                *               |               *
                **              |              ** <- x^(2*power) is the shape of the cost curve
                 ***            |            ***
                   ***          |          ***
             ---------*********************---------
                                |         |<-->| dist_zero_to_nominal

    Args:
        x: The point at which we want to evaluate the barrier function.
        x_nominal: x-value of the point at which the error is equal to error_nominal.
        error_nominal: Error returned when x equals x_nominal.
        dist_zero_to_nominal: Distance from x_nominal to the closest point at which the error is
            zero. Note that dist_zero_to_nominal must be less than x_nominal and greater than zero.
        power: The power used to describe the curve of the error tails.
    """
    return max_power_barrier(
        x=sf.Abs(x),
        x_nominal=x_nominal,
        error_nominal=error_nominal,
        dist_zero_to_nominal=dist_zero_to_nominal,
        power=power,
    )


def min_max_power_barrier(
    x: sf.Scalar,
    x_nominal_lower: sf.Scalar,
    x_nominal_upper: sf.Scalar,
    error_nominal: sf.Scalar,
    dist_zero_to_nominal: sf.Scalar,
    power: sf.Scalar,
) -> sf.Scalar:
    """
    A symmetric barrier centered between x_nominal_lower and x_nominal_upper. See
    symmetric_power_barrier for a detailed description of the barrier function.
    As an example, the barrier with power = 1 will look like:

                                     **          |              **
                                      **         |             **
    (x_nominal_lower, error_nominal) - **        |            ** - (x_nominal_upper, error_nominal)
                                        **       |           **
                                         **      |          ** <- x^power is the shape of the curve
                                          **     |         **
                                 ----------*****************---------
                  dist_zero_to_nominal |<->|     |         |<->| dist_zero_to_nominal

    Args:
        x: The point at which we want to evaluate the barrier function.
        x_nominal_lower: x-value of the point at which the error is equal to error_nominal on
            the left-hand side of the barrier function.
        x_nominal_upper: x-value of the point at which the error is equal to error_nominal on
            the right-hand side of the barrier function.
        error_nominal: Error returned when x equals x_nominal_lower or x_nominal_upper.
        dist_zero_to_nominal: The distance from either of the x_nominal points to the region of
            zero error. Must be less than half the distance between x_nominal_lower and
            x_nominal_upper, and must be greater than zero.
        power: The power used to describe the curve of the error tails. Note that
            when applying the barrier function to a residual used in a least-squares problem,
            a power = 1 will lead to a quadratic cost in the optimization problem.
    """
    center = (x_nominal_lower + x_nominal_upper) / 2
    x_shifted = x - center
    x_nominal_shifted = x_nominal_upper - center
    return symmetric_power_barrier(
        x=x_shifted,
        x_nominal=x_nominal_shifted,
        error_nominal=error_nominal,
        dist_zero_to_nominal=dist_zero_to_nominal,
        power=power,
    )


def min_max_linear_barrier(
    x: sf.Scalar,
    x_nominal_lower: sf.Scalar,
    x_nominal_upper: sf.Scalar,
    error_nominal: sf.Scalar,
    dist_zero_to_nominal: sf.Scalar,
) -> sf.Scalar:
    """
    Applies "min_max_power_barrier" with power = 1. When applied to a residual of a least-squares
    problem, this produces a quadratic cost in the optimization problem because
    cost = 1/2 * residual^2. See "min_max_power_barrier" for more details.
    """
    return min_max_power_barrier(
        x=x,
        x_nominal_lower=x_nominal_lower,
        x_nominal_upper=x_nominal_upper,
        error_nominal=error_nominal,
        dist_zero_to_nominal=dist_zero_to_nominal,
        power=1,
    )


def min_max_centering_power_barrier(
    x: sf.Scalar,
    x_nominal_lower: sf.Scalar,
    x_nominal_upper: sf.Scalar,
    error_nominal: sf.Scalar,
    dist_zero_to_nominal: sf.Scalar,
    power: sf.Scalar,
    centering_scale: sf.Scalar,
) -> sf.Scalar:
    """
    This barrier is the maximum of two power barriers which we call the "bounding" barrier
    and the "centering" barrier. Both barriers are centered between x_nominal_lower and
    x_nominal_upper. As an example, the barrier with power = 1 may look like:

    BARRIER (max of bounding and centering barriers):
                   **              |                          **
                    ** <-(x_nominal_lower, error_nominal)    ** <-(x_nominal_upper, error_nominal)
                     **            |                        **
                      **           |                       **
                       ******      |                  ******
                            ****** |             ****** <- x^power is the shape of upper/lower curve
                                 ******     ******
                    ------------------*******-------------------
                                   |

    It may be easier to vizualize the bounding and centering barriers independently:

    BOUNDING BARRIER:
                   **              |                          **
                    ** <-(x_nominal_lower, error_nominal)    ** <-(x_nominal_upper, error_nominal)
                     **            |                        **
                      **           |                       **
                       **          |                      ** <- x^power is the shape of the curve
                        **         |                     **
                         **        |                    **
                    ------*******************************-------
                                   |                   |<-->| dist_zero_to_nominal

    CENTERING BARRIER:
                                   |
                                   |
             ******                |                            ******
                  ******           |                       ******
       nominal_lower ^ ******      |                  ****** ^ nominal_upper
                            ****** |             ******
                                 ******     ******  <- x^power is the shape of the curve
                    ------------------*******-------------------
                                   |     ^-((x_nominal_lower + x_nominal_upper) / 2, 0)
    where:
        nominal_lower = (x_nominal_lower, centering_scale * error_nominal)
        nominal_upper = (x_nominal_upper, centering_scale * error_nominal)
    and the only point with zero error is the midpoint of x_nominal_lower and x_nominal_upper.

    Args:
        x: The point at which we want to evaluate the barrier function.
        x_nominal_lower: x-value of the point at which the error is equal to error_nominal on
            the left-hand side of the barrier function.
        x_nominal_upper: x-value of the point at which the error is equal to error_nominal on
            the right-hand side of the barrier function.
        error_nominal: Error returned when x equals x_nominal_lower or x_nominal_upper.
        dist_zero_to_nominal: Used with the "bounding barrier" to define the distance from either
            of the x_nominal points to the region of zero error. Must be less than half the
            distance between x_nominal_lower and x_nominal_upper, and must be greater than zero.
        power: The power used to describe the curve of the error tails. Note that
            when applying the barrier function to a residual used in a least-squares problem,
            a power = 1 will lead to a quadratic cost in the optimization problem.
        centering_scale: Used to define the shape of the "centering barrier". Must be between
            zero and one. The centering barrier passes through (x_nominal_lower,
            centering_scale * error_nominal), ((x_nominal_lower + x_nominal_upper) / 2, 0),
            and (x_nominal_upper, centering_scale * error_nominal).
    """
    bounding_barrier = min_max_power_barrier(
        x=x,
        x_nominal_lower=x_nominal_lower,
        x_nominal_upper=x_nominal_upper,
        error_nominal=error_nominal,
        dist_zero_to_nominal=dist_zero_to_nominal,
        power=power,
    )
    center = (x_nominal_lower + x_nominal_upper) / 2
    centering_barrier = min_max_power_barrier(
        x=x,
        x_nominal_lower=x_nominal_lower,
        x_nominal_upper=x_nominal_upper,
        error_nominal=centering_scale * error_nominal,
        dist_zero_to_nominal=x_nominal_upper - center,
        power=power,
    )
    return sf.Max(bounding_barrier, centering_barrier)
