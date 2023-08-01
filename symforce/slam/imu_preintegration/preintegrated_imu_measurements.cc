/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./preintegrated_imu_measurements.h"

namespace sym {

template <typename Scalar>
typename PreintegratedImuMeasurements<Scalar>::Delta
PreintegratedImuMeasurements<Scalar>::Delta::FromLcm(
    const imu_integrated_measurement_delta_t& msg) {
  return {static_cast<Scalar>(msg.Dt), Rot3{msg.DR.cast<Scalar>()}, Vector3{msg.Dv.cast<Scalar>()},
          Vector3{msg.Dp.cast<Scalar>()}};
}

template <typename Scalar>
imu_integrated_measurement_delta_t PreintegratedImuMeasurements<Scalar>::Delta::GetLcmType() const {
  return {static_cast<double>(Dt), DR.Quaternion().template cast<double>(),
          Dv.template cast<double>(), Dp.template cast<double>()};
}

template <typename Scalar>
PreintegratedImuMeasurements<Scalar> PreintegratedImuMeasurements<Scalar>::FromLcm(
    const imu_integrated_measurement_t& msg) {
  PreintegratedImuMeasurements<Scalar> integrated_imu_measurement{
      msg.biases.accel_bias.cast<Scalar>(), msg.biases.gyro_bias.cast<Scalar>()};
  integrated_imu_measurement.delta = Delta::FromLcm(msg.delta);
  integrated_imu_measurement.DR_D_gyro_bias = msg.derivatives.DR_D_gyro_bias.cast<Scalar>();
  integrated_imu_measurement.Dv_D_accel_bias = msg.derivatives.Dv_D_accel_bias.cast<Scalar>();
  integrated_imu_measurement.Dv_D_gyro_bias = msg.derivatives.Dv_D_gyro_bias.cast<Scalar>();
  integrated_imu_measurement.Dp_D_accel_bias = msg.derivatives.Dp_D_accel_bias.cast<Scalar>();
  integrated_imu_measurement.Dp_D_gyro_bias = msg.derivatives.Dp_D_gyro_bias.cast<Scalar>();
  return integrated_imu_measurement;
}

template <typename Scalar>
PreintegratedImuMeasurements<Scalar>::PreintegratedImuMeasurements(const Vector3& accel_bias,
                                                                   const Vector3& gyro_bias)
    : accel_bias{accel_bias},
      gyro_bias{gyro_bias},
      delta{},
      DR_D_gyro_bias{Matrix33::Zero()},
      Dv_D_accel_bias{Matrix33::Zero()},
      Dv_D_gyro_bias{Matrix33::Zero()},
      Dp_D_accel_bias{Matrix33::Zero()},
      Dp_D_gyro_bias{Matrix33::Zero()} {}

template <typename Scalar>
typename PreintegratedImuMeasurements<Scalar>::Delta
PreintegratedImuMeasurements<Scalar>::GetBiasCorrectedDelta(const Vector3& new_accel_bias,
                                                            const Vector3& new_gyro_bias) const {
  const Vector3 accel_bias_delta = new_accel_bias - accel_bias;
  const Vector3 gyro_bias_delta = new_gyro_bias - gyro_bias;

  return {delta.Dt, delta.DR.Retract(DR_D_gyro_bias * gyro_bias_delta),
          delta.Dv + Dv_D_accel_bias * accel_bias_delta + Dv_D_gyro_bias * gyro_bias_delta,
          delta.Dp + Dp_D_accel_bias * accel_bias_delta + Dp_D_gyro_bias * gyro_bias_delta};
}

template <typename Scalar>
imu_integrated_measurement_t PreintegratedImuMeasurements<Scalar>::GetLcmType() const {
  imu_integrated_measurement_t msg;
  msg.biases.accel_bias = accel_bias.template cast<double>();
  msg.biases.gyro_bias = gyro_bias.template cast<double>();
  msg.delta = delta.GetLcmType();
  msg.derivatives.DR_D_gyro_bias = DR_D_gyro_bias.template cast<double>();
  msg.derivatives.Dv_D_accel_bias = Dv_D_accel_bias.template cast<double>();
  msg.derivatives.Dv_D_gyro_bias = Dv_D_gyro_bias.template cast<double>();
  msg.derivatives.Dp_D_accel_bias = Dp_D_accel_bias.template cast<double>();
  msg.derivatives.Dp_D_gyro_bias = Dp_D_gyro_bias.template cast<double>();
  return msg;
}

}  // namespace sym

template struct sym::PreintegratedImuMeasurements<double>;
template struct sym::PreintegratedImuMeasurements<float>;
