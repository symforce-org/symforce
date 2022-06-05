# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce.symbolic as sf
from symforce import typing as T


def imu_preintegration_update(
    # Initial state
    state: sf.V9,
    state_cov: sf.M99,
    state_D_accel_bias: sf.M93,
    state_D_gyro_bias: sf.M93,
    # Biases and noise model
    accel_bias: sf.V3,
    gyro_bias: sf.V3,
    accel_cov: sf.M33,
    gyro_cov: sf.M33,
    # Measurement
    accel: sf.V3,
    gyro: sf.V3,
    dt: T.Scalar,
    # Singularity handling
    epsilon: T.Scalar,
) -> T.Tuple[sf.V9, sf.Matrix, sf.Matrix, sf.Matrix]:
    """
    Symbolic function to perform the full update of incorporating a new IMU measurement
    into the state of a preintegrated IMU factor.
    """
    # Correct for IMU bias
    corrected_accel = accel - accel_bias
    corrected_gyro = gyro - gyro_bias

    # Integrate the state
    new_state = integrate_state(
        state, accel=corrected_accel, gyro=corrected_gyro, dt=dt, epsilon=epsilon
    )

    # Compute jacobians
    new_D_old = new_state.jacobian(state)
    new_D_accel = new_state.jacobian(corrected_accel)
    new_D_gyro = new_state.jacobian(corrected_gyro)
    accel_D_bias = corrected_accel.jacobian(accel_bias)
    gyro_D_bias = corrected_gyro.jacobian(gyro_bias)

    # Update the state covariance
    new_state_cov = new_D_old * state_cov * new_D_old.T
    new_state_cov += new_D_accel * (accel_cov / dt) * new_D_accel.T
    new_state_cov += new_D_gyro * (gyro_cov / dt) * new_D_gyro.T

    # Update the bias derivatives
    new_state_D_accel_bias = new_D_old * state_D_accel_bias + new_D_accel * accel_D_bias
    new_state_D_gyro_bias = new_D_old * state_D_gyro_bias + new_D_gyro * gyro_D_bias

    return new_state, new_state_cov, new_state_D_accel_bias, new_state_D_gyro_bias


def integrate_state(
    state: sf.V9, accel: sf.V3, gyro: sf.V3, dt: T.Scalar, epsilon: T.Scalar
) -> sf.V9:
    """
    Update the given preintegrated state with the given IMU measurement in body frame.
    """
    w_ij, p_ij, v_ij = state[0:3, 0], state[3:6, 0], state[6:9, 0]
    dt22 = 0.5 * dt * sf.Abs(dt)

    R_ij = sf.Rot3.from_tangent(w_ij, epsilon=epsilon)
    R_ik = R_ij.retract(gyro * dt, epsilon=epsilon)

    # Convert acceleration to frame (i)
    # Average over the interval [j, k] for increased accuracy. This measurably reduces
    # integration error over longer intervals, but does not match the GTSAM implementation.
    average_accel = False
    if average_accel:
        a_i = (R_ij * accel + R_ik * accel) / 2
    else:
        a_i = R_ij * accel

    # Create the integrated result
    new_state = sf.Vector9.zero()
    new_state[0:3] = sf.M(R_ik.to_tangent(epsilon=epsilon))
    new_state[3:6] = p_ij + (v_ij * dt) + (a_i * dt22)
    new_state[6:9] = v_ij + (a_i * dt)

    return new_state
