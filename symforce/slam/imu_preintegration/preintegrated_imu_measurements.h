/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <Eigen/Core>

#include <sym/rot3.h>

namespace sym {

/**
 * Struct of Preintegrated IMU Measurements (not including the covariance of change in
 * orientation, velocity, and position).
 */
template <typename Scalar>
struct PreintegratedImuMeasurements {
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using Matrix33 = Eigen::Matrix<Scalar, 3, 3>;

  // The rotation that occured over the measurement period; i.e., maps the coordinates of a vector
  // in the body frame of the end of the measurement period to the coordinates of the vector in the
  // body frame at the start of the measurement period.
  sym::Rot3<Scalar> DR;

  // The velocity change that occured over the measurement period in the body frame of the
  // initial measurement (assuming 0 acceleration due to gravity)
  Vector3 Dv;

  // The position change that occured over the measurement period in the body frame of the
  // initial measurement (assuming 0 acceleration due to gravity and 0 initial velocity)
  Vector3 Dp;

  // Derivatives of DR/Dv/Dp w.r.t. the gyroscope/accelerometer bias linearized at the values
  // of gyro_bias and accel_bias
  Matrix33 DR_D_gyro_bias;
  Matrix33 Dv_D_accel_bias;
  Matrix33 Dv_D_gyro_bias;
  Matrix33 Dp_D_accel_bias;
  Matrix33 Dp_D_gyro_bias;

  // The assumed accelometer bias used during preintegration
  Vector3 accel_bias;

  // The assumed gyroscope bias used during preintegration
  Vector3 gyro_bias;

  // The elapsed time of the measurement period
  Scalar integrated_dt;

  /**
   * Initialize instance struct with accel_bias and gyro_bias and all other values
   * zeroed out (scalars, vectors, and matrices) or set to the identity (DR).
   */
  PreintegratedImuMeasurements(const Vector3& accel_bias, const Vector3& gyro_bias);
};

using PreintegratedImuMeasurementsd = PreintegratedImuMeasurements<double>;
using PreintegratedImuMeasurementsf = PreintegratedImuMeasurements<float>;

}  // namespace sym

extern template struct sym::PreintegratedImuMeasurements<double>;
extern template struct sym::PreintegratedImuMeasurements<float>;
