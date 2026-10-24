# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce.symbolic as sf
from symforce import typing as T
from symforce.jacobian_helpers import tangent_jacobians


def roll_forward_state(
    pose_i: sf.Pose3,
    vel_i: sf.V3,
    # Preintegrated measurements: state
    DR: sf.Rot3,
    Dv: sf.V3,
    Dp: sf.V3,
    # other
    gravity: sf.V3,
    dt: T.Scalar,
) -> T.Tuple[sf.Pose3, sf.V3]:
    """
    Roll forward the given state by the given preintegrated measurements

    This returns the pose_j and vel_j that give 0 residual

    Args:
        pose_i: Pose at time step i (world_T_body)
        vel_i: Velocity at time step i (world frame)
        DR: Preintegrated estimate for pose_i.R.inverse() * pose_j.R
        Dv: Preintegrated estimate for vel_j - vel_i in the body frame at timestep i
        Dp: Preintegrated estimate for position change (before velocity and gravity
        gravity: Acceleration due to gravity (in the same frame as pose_x and vel_x), i.e., the
            vector which when added to the accelerometer measurements gives the true acceleration
            (up to bias and noise) of the IMU.
        dt: The time between timestep i and j: t_j - t_i

    Returns:
        T.Tuple[sf.Pose3, sf.V3]: pose_j, vel_j
    """
    # NOTE(aaron): Derived by setting the residual below to 0 and solving for pose_j and vel_j
    return (
        sf.Pose3(
            R=pose_i.R * DR,
            t=pose_i.R * Dp + pose_i.t + vel_i * dt + sf.S(1) / 2 * gravity * dt**2,
        ),
        pose_i.R * Dv + vel_i + gravity * dt,
    )


def internal_imu_residual(  # noqa: PLR0913, PLR0917
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
    the one found in ``symforce/slam/imu_preintegration/imu_factor.h``.

    Time step i is the time of the first IMU measurement of the interval.
    Time step j is the time after the last IMU measurement of the interval.

    Assumes sqrt_info is lower triangular and only reads the lower triangle.

    Args:
        pose_i: Pose at time step i (world_T_body)
        vel_i: Velocity at time step i (world frame)
        pose_j: Pose at time step j (world_T_body)
        vel_j: Velocity at time step j (world frame)
        accel_bias_i: The accelerometer bias between timesteps i and j
        gyro_bias_i: The gyroscope bias between timesteps i and j
        DR: Preintegrated estimate for pose_i.R.inverse() * pose_j.R
        Dv: Preintegrated estimate for vel_j - vel_i in the body frame at timestep i
        Dp: Preintegrated estimate for position change (before velocity and gravity corrections) in
            the body frame at timestep i::

                R_i^T (p_j - p_i - v_i Delta t - 1/2 g Delta t^2),
                where R_i = pose_i.R, p_i = pose_i.t, p_j = pose_j.t, and v_i = vel_i
        sqrt_info: sqrt info matrix of DR('s tangent space), Dv, Dp
        DR_D_gyro_bias: Derivative of DR w.r.t. gyro_bias evaluated at gyro_bias_hat
        Dv_D_accel_bias: Derivative of Dv w.r.t. accel_bias evaluated at accel_bias_hat
        Dv_D_gyro_bias: Derivative of Dv w.r.t. gyro_bias evaluated at gyro_bias_hat
        Dp_D_accel_bias: Derivative of Dp w.r.t. accel_bias evaluated at accel_bias_hat
        Dp_D_gyro_bias: Derivative of Dp w.r.t. gyro_bias evaluated at gyro_bias_hat
        accel_bias_hat: The accelerometer bias used during preintegration
        gyro_bias_hat: The gyroscope bias used during preintegration
        gravity: Acceleration due to gravity (in the same frame as pose_x and vel_x), i.e., the
            vector which when added to the accelerometer measurements gives the true acceleration
            (up to bias and noise) of the IMU.
        dt: The time between timestep i and j: t_j - t_i
        epsilon: epsilon used for numerical stability

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
        pose_i.R.inverse() * (pose_j.t - pose_i.t - vel_i * dt - sf.S(1) / 2 * gravity * dt**2)
        - corrected_Dp
    )

    res = sf.V9(*res_R, *res_v, *res_p)

    return sf.M91(sqrt_info.lower_triangle() * res)


def _right_jacobian(tangent: sf.V3, epsilon: T.Scalar) -> sf.M33:
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
    out -= ((1 - sf.cos(norm)) / (norm**2)) * tangent_hat
    out += ((norm - sf.sin(norm)) / (norm**3)) * (tangent_hat * tangent_hat)
    return out


def handwritten_new_state_D_state_gyro_accel(
    DR: sf.Rot3, corrected_gyro: sf.V3, corrected_accel: sf.V3, dt: T.Scalar, epsilon: T.Scalar
) -> T.Tuple[sf.M99, sf.M93, sf.M93]:
    """
    Calculates the derivatives of the new state (meaning the DR, Dv, and Dp) w.r.t. the previous state,
    gyroscope measurement, and the accelerometer measurement.

    Args:
        DR: Preintegrated DR of the previous state
        corrected_gyro: gyroscope measurment corrected for IMU bias
        corrected_accel: accelerometer measurement corrected for IMU bias
        dt: Time difference between the previous time step and the new time step
        epsilon: epsilon for numerical stability

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
    new_DR_D_gyro = _right_jacobian(corrected_gyro * dt, epsilon) * dt
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


def step_noise_covariance(
    DR: sf.Rot3,
    corrected_accel: sf.V3,
    accel_cov_diagonal: sf.V3,
    gyro_cov_diagonal: sf.V3,
    dt: T.Scalar,
) -> sf.M99:
    """
    Calculates the covariance of the error that the gyroscope and accelerometer noise of a single
    integration step adds to the new state (DR, Dv, Dp), in local coordinates of the new state.

    The noise is modeled as continuous-time white noise with the given spectral densities, acting
    over the whole step, while the integrator holds the measurements and the rotation DR of the
    start of the step constant (see :func:`integrate_state`). The linearized error state
    ``(dR, dv, dp)`` then evolves within the step as::

        d/dt [dR, dv, dp] = F [dR, dv, dp] + G [gyro_noise, accel_noise]

        F = [[0, 0, 0],      G = [[I, 0],      A = -R [corrected_accel]_x
             [A, 0, 0],           [0, R],
             [0, I, 0]]           [0, 0]]

    where ``R`` is the rotation matrix of DR. ``F`` is nilpotent (``F^3 = 0``), so the state
    transition matrix over a time ``s`` is exactly ``Phi(s) = I + F s + F^2 s^2 / 2``, and the
    covariance of the noise integrated over the step is, exactly::

        Q = int_0^dt Phi(dt - tau) G Sigma G^T Phi(dt - tau)^T dtau
          = sum_{i, j in 0..2} F^i (G Sigma G^T) (F^T)^j dt^(i + j + 1) / (i! j! (i + j + 1))

    Written out by blocks, with ``Sg = diag(gyro_cov_diagonal)`` and
    ``Sa = R diag(accel_cov_diagonal) R^T``::

        Q_RR = Sg dt
        Q_vR = A Sg dt^2 / 2
        Q_pR = A Sg dt^3 / 6
        Q_vv = Sa dt       + A Sg A^T dt^3 / 3
        Q_pv = Sa dt^2 / 2 + A Sg A^T dt^4 / 8
        Q_pp = Sa dt^3 / 3 + A Sg A^T dt^5 / 20

    The ``Sa`` terms are the familiar covariance of doubly integrated white noise, which is
    positive definite for any ``dt > 0``. The ``A Sg`` terms are the rotation noise accumulated
    during the step leaking into the velocity and position through the integrated acceleration.

    This is exact for a non-rotating body. Holding DR fixed within the step drops the
    ``-[omega]_x dR`` term of the rotation-error dynamics (and the right Jacobian ``J_r(omega dt)``
    the discrete model applied to the gyro noise), so with an angular rate ``omega`` the step
    covariance is accurate to first order in ``|omega| dt``. Likewise, the covariance of a window
    does not depend on how it is divided into steps when the body does not rotate, and depends on
    it only to that order otherwise.

    Args:
        DR: Preintegrated DR at the start of the step
        corrected_accel: Accelerometer measurement corrected for IMU bias
        accel_cov_diagonal: Diagonal of the accelerometer noise spectral density
        gyro_cov_diagonal: Diagonal of the gyroscope noise spectral density
        dt: Duration of the step

    Returns:
        sf.M99: The noise covariance to add to the propagated covariance of the new state
    """
    R = DR.to_rotation_matrix()
    A = -R * sf.Matrix.skew_symmetric(corrected_accel)

    gyro_cov = sf.M33.diag(gyro_cov_diagonal.to_flat_list())
    rotated_accel_cov = R * sf.M33.diag(accel_cov_diagonal.to_flat_list()) * R.T
    A_gyro_cov = A * gyro_cov
    A_gyro_cov_AT = A_gyro_cov * A.T

    Q_RR = gyro_cov * dt
    Q_vR = A_gyro_cov * (dt**2 / 2)
    Q_pR = A_gyro_cov * (dt**3 / 6)
    Q_vv = rotated_accel_cov * dt + A_gyro_cov_AT * (dt**3 / 3)
    Q_pv = rotated_accel_cov * (dt**2 / 2) + A_gyro_cov_AT * (dt**4 / 8)
    Q_pp = rotated_accel_cov * (dt**3 / 3) + A_gyro_cov_AT * (dt**5 / 20)

    return sf.M99(
        sf.Matrix.block_matrix(
            [
                [Q_RR, Q_vR.T, Q_pR.T],
                [Q_vR, Q_vv, Q_pv.T],
                [Q_pR, Q_pv, Q_pp],
            ]
        )
    )


def imu_manifold_preintegration_update(  # noqa: PLR0913, PLR0917
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
    ``IntegrateMeasurement`` method of the ``PreintegratedImuMeasurements`` class located in
    ``symforce/slam/imu_preintegration/preintegrated_imu_measurements.h``.

    When integrating the first measurement, DR should be the identity, Dv, Dp, covariance, and the
    derivatives w.r.t. to bias should all be 0.

    The measurement noise is modeled as continuous-time white noise acting over the whole step,
    and its contribution to the covariance is integrated in closed form over the step, holding
    the step's starting rotation fixed (see :func:`step_noise_covariance`). The covariance of a
    window of measurements is therefore full rank even when the window consists of a single step.
    It is exact, and does not depend on how the window is divided into steps, for a non-rotating
    body; with an angular rate it is accurate to first order in the rotation per step.

    Args:
        DR: Preintegrated change in orientation
        Dv: Preintegrated change in velocity
        Dp: Preintegrated change in position
        covariance: Covariance [DR, Dv, Dp] in local coordinates of mean
        DR_D_gyro_bias: Derivative of DR w.r.t. gyro_bias
        Dv_D_accel_bias: Derivative of Dv w.r.t. accel_bias
        Dv_D_gyro_bias: Derivative of Dv w.r.t. gyro_bias
        Dp_D_accel_bias: Derivative of Dp w.r.t. accel_bias
        Dp_D_gyro_bias: Derivative of Dp w.r.t. gyro_bias
        accel_bias: Initial guess for the accelerometer measurement bias
        gyro_bias: Initial guess for the gyroscope measurement bias
        accel_cov_diagonal: The diagonal of the noise spectral density of the accelerometer
            measurement, i.e. the covariance of a measurement times the time it covers
        gyro_cov_diagonal: The diagonal of the noise spectral density of the gyroscope
            measurement, i.e. the covariance of a measurement times the time it covers
        accel_measurement: The accelerometer measurement
        gyro_measurement: The gyroscope measurement
        dt: The time between the new measurements and the last
        epsilon: epsilon for numerical stability

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
                    tangent_jacobians(new_DR, wrt_variables, epsilon),
                    tangent_jacobians(new_Dv, wrt_variables, epsilon),
                    tangent_jacobians(new_Dp, wrt_variables, epsilon),
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

    covariance = covariance.symmetric_copy(sf.Matrix.Triangle.LOWER)

    # The noise of the step is modeled as white noise acting over the whole step, rather than as a
    # perturbation of the held measurement, so its covariance is not expressible through
    # new_state_D_gyro and new_state_D_accel; see step_noise_covariance for the derivation.
    new_covariance = new_state_D_state * covariance * new_state_D_state.T
    new_covariance += step_noise_covariance(
        DR, corrected_accel, accel_cov_diagonal, gyro_cov_diagonal, dt
    )

    new_covariance = new_covariance.symmetric_copy(sf.Matrix.Triangle.LOWER)

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
    new_Dp = Dp + Dv * dt + DR * accel * dt**2 / 2

    return new_DR, new_Dv, new_Dp
