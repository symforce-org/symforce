# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce.symbolic as sf
from symforce import typing as T
from symforce.jacobian_helpers import tangent_jacobians


def internal_imu_residual(
    pose_i: sf.Pose3,
    vel_i: sf.V3,
    pose_j: sf.Pose3,
    vel_j: sf.V3,
    accel_bias_i: sf.V3,
    gyro_bias_i: sf.V3,
    # Preintegrated measurements: state
    DR: sf.Rot3,
    Dv: sf.V3,
    Dp: sf.V3,
    # Other pre-integrated quantities
    sqrt_info: sf.M99,
    DR_D_gyro_bias: sf.M33,
    Dv_D_accel_bias: sf.M33,
    Dv_D_gyro_bias: sf.M33,
    Dp_D_accel_bias: sf.M33,
    Dp_D_gyro_bias: sf.M33,
    # other
    accel_bias_hat: sf.V3,
    gyro_bias_hat: sf.V3,
    gravity: sf.V3,
    dt: T.Scalar,
    epsilon: T.Scalar,
) -> sf.V9:
    """
    An internal helper function to linearize the difference between the orientation, velocity, and
    position at one time step and those of another time step. The expected difference is calculated
    from the preintegrated IMU measurements between those time steps.

    NOTE: If you are looking for a residual for an IMU factor, do not use this. Instead use
    the one found in symforce/slam/imu_preintegration/imu_factor.h.

    Time step i is the time of the first IMU measurement of the interval.
    Time step j is the time after the last IMU measurement of the interval.

    Assumes sqrt_info is lower triangular and only reads the lower triangle.

    Args:
        pose_i (sf.Pose3): Pose at time step i (world_T_body)
        vel_i (sf.V3): Velocity at time step i (world frame)
        pose_j (sf.Pose3): Pose at time step j (world_T_body)
        vel_j (sf.V3): Velocity at time step j (world frame)
        accel_bias_i (sf.V3): The accelerometer bias between timesteps i and j
        gyro_bias_i (sf.V3): The gyroscope bias between timesteps i and j
        DR (sf.Rot3): Preintegrated estimate for pose_i.R.inverse() * pose_j.R
        Dv (sf.V3): Preintegrated estimate for vel_j - vel_i in the body frame at timestep i
        Dp (sf.V3): Preintegrated estimate for position change (before velocity and gravity
            corrections) in the body frame at timestep i:
            R_i^T (p_j - p_i - v_i Delta t - 1/2 g Delta t^2),
            where R_i = pose_i.R, p_i = pose_i.t, p_j = pose_j.t, and v_i = vel_i
        sqrt_info (sf.M99): sqrt info matrix of DR('s tangent space), Dv, Dp
        DR_D_gyro_bias (sf.M33): Derivative of DR w.r.t. gyro_bias evaluated at gyro_bias_hat
        Dv_D_accel_bias (sf.M33): Derivative of Dv w.r.t. accel_bias evaluated at accel_bias_hat
        Dv_D_gyro_bias (sf.M33): Derivative of Dv w.r.t. gyro_bias evaluated at gyro_bias_hat
        Dp_D_accel_bias (sf.M33): Derivative of Dp w.r.t. accel_bias evaluated at accel_bias_hat
        Dp_D_gyro_bias (sf.M33): Derivative of Dp w.r.t. gyro_bias evaluated at gyro_bias_hat
        accel_bias_hat (sf.V3): The accelerometer bias used during preintegration
        gyro_bias_hat (sf.V3): The gyroscope bias used during preintegration
        gravity (sf.V3): Acceleration due to gravity (in the same frame as pose_x and vel_x),
            i.e., the vector which when added to the accelerometer measurements gives the
            true acceleration (up to bias and noise) of the IMU.
        dt (T.Scalar): The time between timestep i and j: t_j - t_i
        epsilon (T.Scalar): epsilon used for numerical stability

    Outputs:
        res: 9dof product residual between the orientations, then velocities, then positions.
    """
    delta_gyro_bias = gyro_bias_i - gyro_bias_hat
    delta_accel_bias = accel_bias_i - accel_bias_hat

    # Correct preintegrated measurements for updated bias estimates
    corrected_DR = DR * sf.Rot3.from_tangent(
        (DR_D_gyro_bias * delta_gyro_bias).to_storage(), epsilon
    )
    corrected_Dv = Dv + Dv_D_gyro_bias * delta_gyro_bias + Dv_D_accel_bias * delta_accel_bias
    corrected_Dp = Dp + Dp_D_gyro_bias * delta_gyro_bias + Dp_D_accel_bias * delta_accel_bias

    res_R = (corrected_DR.inverse() * pose_i.R.inverse() * pose_j.R).to_tangent(epsilon)
    res_v = pose_i.R.inverse() * (vel_j - vel_i - gravity * dt) - corrected_Dv
    res_p = (
        pose_i.R.inverse() * (pose_j.t - pose_i.t - vel_i * dt - sf.S(1) / 2 * gravity * dt ** 2)
        - corrected_Dp
    )

    res = sf.V9(*res_R, *res_v, *res_p)

    return sf.M91(sqrt_info.lower_triangle() * res)


def right_jacobian(tangent: sf.V3, epsilon: T.Scalar) -> sf.M33:
    """
    The right jacobian J(v) is d/du Log(Exp(v)^{-1} Exp(v + u)), i.e., a matrix such that
    Exp(v + dv) ~= Exp(v) Exp(J(v) dv), where v and u are tangent vectors of SO(3).

    Returns J(tangent).
    """
    # NOTE(brad): We use sqrt(epsilon) to ensure cos(norm) is far enough from 1 to mitigate
    # the effects of catastrophic cancellation in 1 - cos(norm) below.
    norm = tangent.norm(sf.sqrt(epsilon))
    tangent_hat = sf.Matrix.skew_symmetric(tangent)
    out = sf.M33.eye()
    out -= ((1 - sf.cos(norm)) / (norm ** 2)) * tangent_hat
    out += ((norm - sf.sin(norm)) / (norm ** 3)) * (tangent_hat * tangent_hat)
    return out


def handwritten_new_state_D_state_gyro_accel(
    DR: sf.Rot3, corrected_gyro: sf.V3, corrected_accel: sf.V3, dt: T.Scalar, epsilon: T.Scalar
) -> T.Tuple[sf.M99, sf.M93, sf.M93]:
    """
    Calculates the derivatives of the new state (meaning the DR, Dv, and Dp) w.r.t. the previous state,
    gyroscope measurement, and the accelerometer mesaurement.


    Args:
        DR (sf.Rot3): Preintegrated DR of the previous state
        corrected_gyro (sf.V3): gyroscope measurment corrected for IMU bias
        corrected_accel (sf.V3): accelerometer measurement corrected for IMU bias
        dt (T.Scalar): Time difference between the previous time step and the new time step
        epsilon (T.Scalar): epsilon for numerical stability

    Returns:
        T.Tuple[sf.M99, sf.M93, sf.M93]: new_state_D_old_state, new_state_D_gyro_measurement, new_state_D_accel_measurement
    """
    # calculate new_state_D_old_state
    DR_update = sf.Rot3.from_tangent(corrected_gyro * dt, epsilon)
    new_DR_D_old_DR = DR_update.inverse().to_rotation_matrix()
    new_DR_D_old_Dv = sf.M33.zero()
    new_DR_D_old_Dp = sf.M33.zero()

    new_Dv_D_old_DR = -DR.to_rotation_matrix() * sf.Matrix.skew_symmetric(corrected_accel * dt)
    new_Dv_D_old_Dv = sf.M33.eye()
    new_Dv_D_old_Dp = sf.M33.zero()

    new_Dp_D_old_DR = (
        (-DR.to_rotation_matrix() * sf.Matrix.skew_symmetric(corrected_accel * dt)) * dt / 2
    )
    new_Dp_D_old_Dv = dt * sf.M33.eye()
    new_Dp_D_old_Dp = sf.M33.eye()

    new_state_D_old_state = sf.Matrix.block_matrix(
        [
            [new_DR_D_old_DR, new_DR_D_old_Dv, new_DR_D_old_Dp],
            [new_Dv_D_old_DR, new_Dv_D_old_Dv, new_Dv_D_old_Dp],
            [new_Dp_D_old_DR, new_Dp_D_old_Dv, new_Dp_D_old_Dp],
        ]
    )

    # calculate new_D_gyro
    new_DR_D_gyro = right_jacobian(corrected_gyro * dt, epsilon) * dt
    new_DvDp_D_gyro = sf.M63.zero()

    new_state_D_gyro = sf.Matrix.block_matrix(
        [
            [new_DR_D_gyro],
            [new_DvDp_D_gyro],
        ]
    )

    # calculate new_D_accel
    new_DR_D_accel = sf.M33.zero()
    new_Dv_D_accel = DR.to_rotation_matrix() * dt
    new_Dp_D_accel = DR.to_rotation_matrix() * (dt * dt / 2)

    new_state_D_accel = sf.Matrix.block_matrix(
        [
            [new_DR_D_accel],
            [new_Dv_D_accel],
            [new_Dp_D_accel],
        ]
    )

    return sf.M99(new_state_D_old_state), sf.M93(new_state_D_gyro), sf.M93(new_state_D_accel)


def imu_manifold_preintegration_update(
    # Initial state
    DR: sf.Rot3,
    Dv: sf.V3,
    Dp: sf.V3,
    covariance: sf.M99,
    DR_D_gyro_bias: sf.M33,
    Dv_D_accel_bias: sf.M33,
    Dv_D_gyro_bias: sf.M33,
    Dp_D_accel_bias: sf.M33,
    Dp_D_gyro_bias: sf.M33,
    # Biases and noise model
    accel_bias: sf.V3,
    gyro_bias: sf.V3,
    accel_cov_diagonal: sf.V3,
    gyro_cov_diagonal: sf.V3,
    # Measurement
    accel_measurement: sf.V3,
    gyro_measurement: sf.V3,
    dt: T.Scalar,
    # Singularity handling,
    epsilon: T.Scalar,
    # Configuration
    use_handwritten_derivatives: bool = True,
) -> T.Tuple[sf.Rot3, sf.V3, sf.V3, sf.M99, sf.M33, sf.M33, sf.M33, sf.M33, sf.M33]:
    """
    An internal helper function to update a set of preintegrated IMU measurements with a new pair of
    gyroscope and accelerometer measurements. Returns the updated preintegrated measurements.

    NOTE: If you are interested in this function, you should likely be using the
    IntegrateMeasurement method of the PreintegratedImuMeasurements class located in
    symforce/slam/imu_preintegration/preintegrated_imu_measurements.h.

    When integrating the first measurement, DR should be the identity, Dv, Dp, covariance, and the
    derivatives w.r.t. to bias should all be 0.

    Args:
        DR (sf.Rot3): Preintegrated change in orientation
        Dv (sf.V3): Preintegrated change in velocity
        Dp (sf.V3): Preintegrated change in position
        covariance (sf.M99): Covariance [DR, Dv, Dp] in local coordinates of mean
        DR_D_gyro_bias (sf.M33): Derivative of DR w.r.t. gyro_bias
        Dv_D_accel_bias (sf.M33): Derivative of Dv w.r.t. accel_bias
        Dv_D_gyro_bias (sf.M33): Derivative of Dv w.r.t. gyro_bias
        Dp_D_accel_bias (sf.M33): Derivative of Dp w.r.t. accel_bias
        Dp_D_gyro_bias (sf.M33): Derivative of Dp w.r.t. gyro_bias
        accel_bias (sf.V3): Initial guess for the accelerometer measurement bias
        gyro_bias (sf.V3): Initial guess for the gyroscope measurement bias
        accel_cov_diagonal (sf.M33): The diagonal of the covariance of the accelerometer measurement
        gyro_cov_diagonal (sf.M33): The diagonal of the covariance of the gyroscope measurement
        accel_measurement (sf.V3): The accelerometer measurement
        gyro_measurement (sf.V3): The gyroscope measurement
        dt (T.Scalar): The time between the new measurements and the last
        epsilon (T.Scalar): epsilon for numerical stability

    Returns:
        T.Tuple[sf.Rot3, sf.V3, sf.V3, sf.M99, sf.M33, sf.M33, sf.M33, sf.M33, sf.M33]:
            new_DR,
            new_Dv,
            new_Dp,
            new_covariance,
            new_DR_D_gyro_bias,
            new_Dv_D_accel_bias,
            new_Dv_D_gyro_bias,
            new_Dp_D_accel_bias
            new_Dp_D_gyro_bias,
    """
    # Correct for IMU bias
    corrected_accel = accel_measurement - accel_bias
    corrected_gyro = gyro_measurement - gyro_bias

    # Integrate the state
    new_DR, new_Dv, new_Dp = integrate_state(
        DR,
        Dv,
        Dp,
        accel=corrected_accel,
        gyro=corrected_gyro,
        dt=dt,
        epsilon=epsilon,
    )

    # NOTE(Brad): Both ways of calculating the derivatives of new_state should be the same.
    if not use_handwritten_derivatives:
        # Definitely correct, but not very amenable to CSE
        def new_state_D(wrt_variables: T.List[T.Any]) -> sf.Matrix:
            return sf.Matrix.block_matrix(
                [
                    tangent_jacobians(new_DR, wrt_variables),
                    tangent_jacobians(new_Dv, wrt_variables),
                    tangent_jacobians(new_Dp, wrt_variables),
                ]
            )

        new_state_D_state = new_state_D([DR, Dv, Dp])
        new_state_D_gyro_bias = new_state_D([gyro_bias])
        new_state_D_accel_bias = new_state_D([accel_bias])

        new_state_D_gyro = new_state_D([gyro_measurement])
        new_state_D_accel = new_state_D([accel_measurement])
    else:
        # Handwritten derivatives, reduces op count by ~20%
        (
            new_state_D_state,
            new_state_D_gyro,
            new_state_D_accel,
        ) = handwritten_new_state_D_state_gyro_accel(
            DR, corrected_gyro, corrected_accel, dt, epsilon
        )
        new_state_D_gyro_bias = -new_state_D_gyro
        new_state_D_accel_bias = -new_state_D_accel

    new_covariance = new_state_D_state * covariance * new_state_D_state.T
    new_covariance += new_state_D_gyro * sf.M.diag(gyro_cov_diagonal / dt) * new_state_D_gyro.T
    new_covariance += new_state_D_accel * sf.M.diag(accel_cov_diagonal / dt) * new_state_D_accel.T

    state_D_bias = sf.Matrix.block_matrix(
        [
            [DR_D_gyro_bias, sf.M33.zero()],
            [Dv_D_gyro_bias, Dv_D_accel_bias],
            [Dp_D_gyro_bias, Dp_D_accel_bias],
        ]
    )

    new_state_D_bias = sf.Matrix.block_matrix([[new_state_D_gyro_bias, new_state_D_accel_bias]])

    new_state_D_bias = new_state_D_state * state_D_bias + new_state_D_bias

    return (
        new_DR,
        new_Dv,
        new_Dp,
        sf.M99(new_covariance),
        sf.M33(new_state_D_bias[0:3, 0:3]),  # new_DR_D_gyro_bias
        sf.M33(new_state_D_bias[3:6, 3:6]),  # new_Dv_D_accel_bias
        sf.M33(new_state_D_bias[3:6, 0:3]),  # new_Dv_D_gyro_bias
        sf.M33(new_state_D_bias[6:9, 3:6]),  # new_Dp_D_accel_bias
        sf.M33(new_state_D_bias[6:9, 0:3]),  # new_Dp_D_gyro_bias
    )


def integrate_state(
    # state
    DR: sf.Rot3,
    Dv: sf.V3,
    Dp: sf.V3,
    accel: sf.V3,
    gyro: sf.V3,
    dt: T.Scalar,
    epsilon: T.Scalar,
) -> T.Tuple[sf.Rot3, sf.V3, sf.V3]:
    """
    Given the old preintegrated state and the new IMU measurements, calculates the
    new preintegrated state.

    Returns:
        T.Tuple[sf.Rot3, sf.V3, sf.V3]: (new_DR, new_Dv, new_Dp)
    """
    new_DR = DR * sf.Rot3.from_tangent(gyro * dt, epsilon)
    new_Dv = Dv + DR * accel * dt
    new_Dp = Dp + Dv * dt + DR * accel * dt ** 2 / 2

    return new_DR, new_Dv, new_Dp
