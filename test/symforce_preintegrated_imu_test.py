# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import numpy as np
from pathlib import Path

import symforce
from symforce import codegen
from symforce import logger
from symforce.path_util import symforce_dir
from symforce.test_util import TestCase, symengine_only

from symforce.slam.imu_preintegration.generate import generate_imu_preintegration
from symforce.slam.imu_preintegration.preintegrated_imu import PreintegratedImu
from symforce.slam.imu_preintegration.test_util import generate_random_imu_measurements


class SymforcePreintegratedImuTest(TestCase):
    """
    Test IMU preintegration.
    """

    @symengine_only
    def test_gen_python(self) -> None:
        assert symforce.get_backend() == "symengine"

        # Generate code
        output_dir = Path(self.make_output_dir("sf_imu_preintegration_test_"))
        generate_imu_preintegration(config=codegen.PythonConfig(), output_dir=output_dir)

        # Compare and update the file
        out_name = output_dir / "imu_preintegration_update.py"
        target_dir = symforce_dir() / "gen" / "python" / "sym" / "factors"
        self.compare_or_update_file(path=target_dir / out_name.name, new_file=out_name)

    @symengine_only
    def test_gen_cpp(self) -> None:
        assert symforce.get_backend() == "symengine"

        # Generate code
        output_dir = Path(self.make_output_dir("sf_imu_preintegration_test_"))
        generate_imu_preintegration(config=codegen.CppConfig(), output_dir=output_dir)

        # Compare and update the file
        out_name = output_dir / "imu_preintegration_update.h"
        target_dir = symforce_dir() / "gen" / "cpp" / "sym" / "factors"
        self.compare_or_update_file(path=target_dir / out_name.name, new_file=out_name)

    def test_python_numerical(self) -> None:
        pim = PreintegratedImu(
            accel_cov=np.eye(3, 3),
            gyro_cov=np.eye(3, 3),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
        )

        imu_measurements = list(generate_random_imu_measurements(num=10))
        for meas in imu_measurements:
            pim.integrate_measurement(accel=meas.accel, gyro=meas.gyro, dt=meas.dt, epsilon=1e-12)

        self.assertEqual(sum(m.dt for m in imu_measurements), pim.delta_t_ij)

        print(f"state=\n{pim.state}")
        print(f"state_cov=\n{pim.state_cov}")
        print(f"state_D_accel_bias=\n{pim.state_D_accel_bias}")
        print(f"state_D_gyro_bias=\n{pim.state_D_gyro_bias}")

    def test_python_numerical_against_gtsam(self) -> None:
        try:
            import gtsam
        except ImportError:
            logger.info("Skipping test, GTSAM not installed.")
            return

        pim_sym = PreintegratedImu(
            accel_cov=np.eye(3, 3),
            gyro_cov=np.eye(3, 3),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
        )

        pim_gtsam = gtsam.PreintegratedImuMeasurements(
            params=gtsam.PreintegrationParams(
                n_gravity=np.array([0, 0, -9.81]),
            ),
            bias=gtsam.gtsam.imuBias.ConstantBias(
                biasAcc=pim_sym.accel_bias,
                biasGyro=pim_sym.gyro_bias,
            ),
        )

        imu_measurements = list(generate_random_imu_measurements(num=10))
        for i, meas in enumerate(imu_measurements):
            pim_sym.integrate_measurement(
                accel=meas.accel, gyro=meas.gyro, dt=meas.dt, epsilon=1e-12
            )

            pim_gtsam.integrateMeasurement(
                measuredAcc=meas.accel,
                measuredOmega=meas.gyro,
                deltaT=meas.dt,
            )

            state_sym = pim_sym.state.T.squeeze()
            state_gtsam = pim_gtsam.preintegrated()

            state_diff_rel = np.linalg.norm(state_sym - state_gtsam) / np.linalg.norm(state_gtsam)
            logger.debug(f"{i}: {state_diff_rel=}")
            self.assertLess(state_diff_rel, 1e-6)


if __name__ == "__main__":
    TestCase.main()
