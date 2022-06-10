# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import numpy as np

from sym.factors.imu_preintegration_update import imu_preintegration_update


class PreintegratedImu:
    """
    Numerical Python class for IMU preintegration.

    Usage example:
        pim = PreintegratedImu(...)

        for meas in imu_measurements:
            pim.integrate_measurement(meas.accel, meas.gyro, meas.dt, epsilon=epsilon)
    """

    def __init__(
        self,
        accel_cov: np.ndarray,
        gyro_cov: np.ndarray,
        accel_bias: np.ndarray,
        gyro_bias: np.ndarray,
    ):
        """
        Args:
            accel_cov:
            gyro_cov:
            accel_bias:
            gyro_bias:
        """
        # IMU measurement white noise covariance
        assert accel_cov.shape == (3, 3)
        self.accel_cov = accel_cov
        assert gyro_cov.shape == (3, 3)
        self.gyro_cov = gyro_cov

        # IMU bias
        assert accel_bias.shape == (3,)
        self.accel_bias = accel_bias.reshape(3, 1)
        assert gyro_bias.shape == (3,)
        self.gyro_bias = gyro_bias.reshape(3, 1)

        # Total time from t(i) to t(j).
        self.delta_t_ij = 0.0

        # Preintegrated state as a 9D vector on tangent space at frame i
        # Order is: theta, position, velocity
        self.state = np.zeros((9, 1))

        # Measurement covariance of the preintegrated state
        self.state_cov = np.zeros((9, 9))

        # Integrated measurements wrt accel and gyro biases.
        self.state_D_accel_bias = np.zeros((9, 3))
        self.state_D_gyro_bias = np.zeros((9, 3))

    def integrate_measurement(
        self, accel: np.ndarray, gyro: np.ndarray, dt: float, epsilon: float
    ) -> None:
        """
        Integrate forward for a single IMU measurement, updating uncertainty and derivatives.

        Args:
            accel:
            gyro:
            dt:
            epsilon:
        """
        assert accel.shape == (3,)
        assert gyro.shape == (3,)

        # Update time
        self.delta_t_ij += dt

        # Update everything else
        (
            self.state,
            self.state_cov,
            self.state_D_accel_bias,
            self.state_D_gyro_bias,
        ) = imu_preintegration_update(
            # State
            state=self.state,
            state_cov=self.state_cov,
            state_D_accel_bias=self.state_D_accel_bias,
            state_D_gyro_bias=self.state_D_gyro_bias,
            # Biases and noise model
            accel_bias=self.accel_bias,
            gyro_bias=self.gyro_bias,
            accel_cov=self.accel_cov,
            gyro_cov=self.gyro_cov,
            # Measurement
            accel=accel.reshape(3, 1),
            gyro=gyro.reshape(3, 1),
            dt=dt,
            # Singularity handling
            epsilon=epsilon,
        )

        self.state = np.array(self.state).reshape(9, 1)
