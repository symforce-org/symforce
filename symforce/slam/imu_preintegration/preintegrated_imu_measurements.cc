/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./preintegrated_imu_measurements.h"

#include <sym/factors/internal/roll_forward_state.h>

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
std::pair<Pose3<Scalar>, Vector3<Scalar>>
PreintegratedImuMeasurements<Scalar>::Delta::RollForwardState(const Pose3<Scalar>& pose_i,
                                                              const Vector3& vel_i,
                                                              const Vector3& gravity) const {
  Pose3<Scalar> pose_j;
  Vector3 vel_j;
  sym::RollForwardState(pose_i, vel_i, DR, Dv, Dp, gravity, Dt, &pose_j, &vel_j);
  return {pose_j, vel_j};
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

// --------------------------------------------------------------------------
// StorageOps concept
// --------------------------------------------------------------------------

template <typename Scalar>
void PreintegratedImuMeasurements<Scalar>::ToStorage(Scalar* const vec) const {
  StorageOps<PreintegratedImuMeasurements>::ToStorage(*this, vec);
}

template <typename Scalar>
PreintegratedImuMeasurements<Scalar> PreintegratedImuMeasurements<Scalar>::FromStorage(
    const Scalar* vec) {
  return StorageOps<PreintegratedImuMeasurements>::FromStorage(vec);
}

template <typename ScalarType>
void StorageOps<PreintegratedImuMeasurements<ScalarType>>::ToStorage(const T& a, ScalarType* out) {
  int idx = 0;

  // Store biases
  StorageOps<Vector3<ScalarType>>::ToStorage(a.accel_bias, out + idx);
  idx += StorageOps<Vector3<ScalarType>>::StorageDim();
  StorageOps<Vector3<ScalarType>>::ToStorage(a.gyro_bias, out + idx);
  idx += StorageOps<Vector3<ScalarType>>::StorageDim();

  // Store delta
  out[idx++] = a.delta.Dt;
  StorageOps<Rot3<ScalarType>>::ToStorage(a.delta.DR, out + idx);
  idx += StorageOps<Rot3<ScalarType>>::StorageDim();
  StorageOps<Vector3<ScalarType>>::ToStorage(a.delta.Dv, out + idx);
  idx += StorageOps<Vector3<ScalarType>>::StorageDim();
  StorageOps<Vector3<ScalarType>>::ToStorage(a.delta.Dp, out + idx);
  idx += StorageOps<Vector3<ScalarType>>::StorageDim();

  // Store derivatives
  StorageOps<Matrix33<ScalarType>>::ToStorage(a.DR_D_gyro_bias, out + idx);
  idx += StorageOps<Matrix33<ScalarType>>::StorageDim();
  StorageOps<Matrix33<ScalarType>>::ToStorage(a.Dv_D_accel_bias, out + idx);
  idx += StorageOps<Matrix33<ScalarType>>::StorageDim();
  StorageOps<Matrix33<ScalarType>>::ToStorage(a.Dv_D_gyro_bias, out + idx);
  idx += StorageOps<Matrix33<ScalarType>>::StorageDim();
  StorageOps<Matrix33<ScalarType>>::ToStorage(a.Dp_D_accel_bias, out + idx);
  idx += StorageOps<Matrix33<ScalarType>>::StorageDim();
  StorageOps<Matrix33<ScalarType>>::ToStorage(a.Dp_D_gyro_bias, out + idx);
}

template <typename ScalarType>
typename StorageOps<PreintegratedImuMeasurements<ScalarType>>::T
StorageOps<PreintegratedImuMeasurements<ScalarType>>::FromStorage(const ScalarType* data) {
  int idx = 0;

  // Load biases
  const Vector3<ScalarType> accel_bias = StorageOps<Vector3<ScalarType>>::FromStorage(data + idx);
  idx += StorageOps<Vector3<ScalarType>>::StorageDim();
  const Vector3<ScalarType> gyro_bias = StorageOps<Vector3<ScalarType>>::FromStorage(data + idx);
  idx += StorageOps<Vector3<ScalarType>>::StorageDim();

  // Create result with biases
  T result(accel_bias, gyro_bias);

  // Load delta
  result.delta.Dt = data[idx++];
  result.delta.DR = StorageOps<Rot3<ScalarType>>::FromStorage(data + idx);
  idx += StorageOps<Rot3<ScalarType>>::StorageDim();
  result.delta.Dv = StorageOps<Vector3<ScalarType>>::FromStorage(data + idx);
  idx += StorageOps<Vector3<ScalarType>>::StorageDim();
  result.delta.Dp = StorageOps<Vector3<ScalarType>>::FromStorage(data + idx);
  idx += StorageOps<Vector3<ScalarType>>::StorageDim();

  // Load derivatives
  result.DR_D_gyro_bias = StorageOps<Matrix33<ScalarType>>::FromStorage(data + idx);
  idx += StorageOps<Matrix33<ScalarType>>::StorageDim();
  result.Dv_D_accel_bias = StorageOps<Matrix33<ScalarType>>::FromStorage(data + idx);
  idx += StorageOps<Matrix33<ScalarType>>::StorageDim();
  result.Dv_D_gyro_bias = StorageOps<Matrix33<ScalarType>>::FromStorage(data + idx);
  idx += StorageOps<Matrix33<ScalarType>>::StorageDim();
  result.Dp_D_accel_bias = StorageOps<Matrix33<ScalarType>>::FromStorage(data + idx);
  idx += StorageOps<Matrix33<ScalarType>>::StorageDim();
  result.Dp_D_gyro_bias = StorageOps<Matrix33<ScalarType>>::FromStorage(data + idx);

  return result;
}

}  // namespace sym

// Explicit instantiation
template struct sym::PreintegratedImuMeasurements<double>;
template struct sym::PreintegratedImuMeasurements<float>;
template struct sym::StorageOps<sym::PreintegratedImuMeasurements<double>>;
template struct sym::StorageOps<sym::PreintegratedImuMeasurements<float>>;
