/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./preintegrated_imu_measurements.h"

namespace sym {

template <typename Scalar>
PreintegratedImuMeasurements<Scalar>::PreintegratedImuMeasurements(const Vector3& accel_bias,
                                                                   const Vector3& gyro_bias)
    : DR(),
      Dv{Vector3::Zero()},
      Dp{Vector3::Zero()},
      DR_D_gyro_bias{Matrix33::Zero()},
      Dv_D_accel_bias{Matrix33::Zero()},
      Dv_D_gyro_bias{Matrix33::Zero()},
      Dp_D_accel_bias{Matrix33::Zero()},
      Dp_D_gyro_bias{Matrix33::Zero()},
      accel_bias{accel_bias},
      gyro_bias{gyro_bias},
      integrated_dt{0.0} {}

template <typename Scalar>
auto PreintegratedImuMeasurements<Scalar>::GetBiasCorrectedDelta(const Vector3& new_accel_bias,
                                                                 const Vector3& new_gyro_bias) const
    -> Delta {
  const Vector3 accel_bias_delta = new_accel_bias - accel_bias;
  const Vector3 gyro_bias_delta = new_gyro_bias - gyro_bias;

  Delta delta;
  delta.dt = integrated_dt;
  delta.DR = DR.Retract(DR_D_gyro_bias * gyro_bias_delta);
  delta.Dv.noalias() = Dv + Dv_D_accel_bias * accel_bias_delta + Dv_D_gyro_bias * gyro_bias_delta;
  delta.Dp.noalias() = Dp + Dp_D_accel_bias * accel_bias_delta + Dp_D_gyro_bias * gyro_bias_delta;
  return delta;
}

}  // namespace sym

template struct sym::PreintegratedImuMeasurements<double>;
template struct sym::PreintegratedImuMeasurements<float>;
