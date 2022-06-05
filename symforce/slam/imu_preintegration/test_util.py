# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import typing as T
from dataclasses import dataclass

import numpy as np


@dataclass
class ImuMeas:
    accel: np.ndarray
    gyro: np.ndarray
    dt: float


def generate_random_imu_measurements(
    num: int,
    accel_sigma: float = 1.0,
    gyro_sigma: float = 0.1,
    min_dt: float = 0.01,
    max_dt: float = 0.25,
) -> T.Iterator[ImuMeas]:
    """
    Generate random IMU measurements from a reasonable distribution.
    """
    for _ in range(num):
        yield ImuMeas(
            accel=np.random.normal(loc=np.array([0, 0, -9.81]), scale=accel_sigma, size=(3,)),
            gyro=np.random.normal(loc=np.array([0, 0, 0]), scale=gyro_sigma, size=(3,)),
            dt=np.random.uniform(low=min_dt, high=max_dt),
        )
