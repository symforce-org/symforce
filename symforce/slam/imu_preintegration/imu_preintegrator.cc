/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./imu_preintegrator.h"

#include <sym/factors/internal/imu_manifold_preintegration_update.h>

namespace sym {

template <typename Scalar>
ImuPreintegrator<Scalar>::ImuPreintegrator(const Vector3& accel_bias, const Vector3& gyro_bias)
    : preintegrated_measurements_(accel_bias, gyro_bias), covariance_{Matrix99::Zero()} {}

template <typename Scalar>
void ImuPreintegrator<Scalar>::IntegrateMeasurement(const Vector3& measured_accel,
                                                    const Vector3& measured_gyro,
                                                    const Vector3& accel_cov,
                                                    const Vector3& gyro_cov, const Scalar dt,
                                                    const Scalar epsilon) {
  auto& delta = preintegrated_measurements_.delta;

  Rot3<Scalar> new_DR;
  Vector3 new_Dv;
  Vector3 new_Dp;
  Matrix99 new_covariance;
  Matrix33 new_DR_D_gyro_bias;
  Matrix33 new_Dv_D_accel_bias;
  Matrix33 new_Dv_D_gyro_bias;
  Matrix33 new_Dp_D_accel_bias;
  Matrix33 new_Dp_D_gyro_bias;

  ImuManifoldPreintegrationUpdate<Scalar>(
      delta.DR, delta.Dv, delta.Dp, covariance_, preintegrated_measurements_.DR_D_gyro_bias,
      preintegrated_measurements_.Dv_D_accel_bias, preintegrated_measurements_.Dv_D_gyro_bias,
      preintegrated_measurements_.Dp_D_accel_bias, preintegrated_measurements_.Dp_D_gyro_bias,
      preintegrated_measurements_.accel_bias, preintegrated_measurements_.gyro_bias, accel_cov,
      gyro_cov, measured_accel, measured_gyro, dt, epsilon,
      // outputs
      &new_DR, &new_Dv, &new_Dp, &new_covariance, &new_DR_D_gyro_bias, &new_Dv_D_accel_bias,
      &new_Dv_D_gyro_bias, &new_Dp_D_accel_bias, &new_Dp_D_gyro_bias);

  delta.Dt += dt;
  delta.DR = new_DR;
  delta.Dv = new_Dv;
  delta.Dp = new_Dp;
  preintegrated_measurements_.DR_D_gyro_bias = new_DR_D_gyro_bias;
  preintegrated_measurements_.Dv_D_accel_bias = new_Dv_D_accel_bias;
  preintegrated_measurements_.Dv_D_gyro_bias = new_Dv_D_gyro_bias;
  preintegrated_measurements_.Dp_D_accel_bias = new_Dp_D_accel_bias;
  preintegrated_measurements_.Dp_D_gyro_bias = new_Dp_D_gyro_bias;
  covariance_ = new_covariance;
}

template <typename Scalar>
const PreintegratedImuMeasurements<Scalar>& ImuPreintegrator<Scalar>::PreintegratedMeasurements()
    const {
  return preintegrated_measurements_;
}

template <typename Scalar>
const typename ImuPreintegrator<Scalar>::Matrix99& ImuPreintegrator<Scalar>::Covariance() const {
  return covariance_;
}

}  // namespace sym

template class sym::ImuPreintegrator<double>;
template class sym::ImuPreintegrator<float>;
