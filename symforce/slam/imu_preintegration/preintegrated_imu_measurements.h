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

  // The rotation that occurred over the measurement period; i.e., maps the coordinates of a vector
  // in the body frame of the end of the measurement period to the coordinates of the vector in the
  // body frame at the start of the measurement period.
  sym::Rot3<Scalar> DR;

  // The velocity change that occurred over the measurement period in the body frame of the
  // initial measurement (assuming 0 acceleration due to gravity)
  Vector3 Dv;

  // The position change that occurred over the measurement period in the body frame of the
  // initial measurement (assuming 0 acceleration due to gravity and 0 initial velocity)
  Vector3 Dp;

  // Derivatives of DR/Dv/Dp w.r.t. the gyroscope/accelerometer bias linearized at the values
  // of gyro_bias and accel_bias
  Matrix33 DR_D_gyro_bias;
  Matrix33 Dv_D_accel_bias;
  Matrix33 Dv_D_gyro_bias;
  Matrix33 Dp_D_accel_bias;
  Matrix33 Dp_D_gyro_bias;

  // The original accelerometer bias used during preintegration
  Vector3 accel_bias;

  // The original gyroscope bias used during preintegration
  Vector3 gyro_bias;

  // The elapsed time of the measurement period
  Scalar integrated_dt;

  // A convenient struct that holds the Preintegrated delta
  struct Delta {
    sym::Rot3<Scalar> DR{};
    Vector3 Dv{Vector3::Zero()};
    Vector3 Dp{Vector3::Zero()};
    Scalar dt{};  // elapsed time
  };

  // Given new accel and gyro biases, return a first-order correction to the preintegrated delta
  // The user is responsible for making sure that the new biases are sufficiently close to the
  // original biases used during the preintegration.
  Delta GetBiasCorrectedDelta(const Vector3& new_accel_bias, const Vector3& new_gyro_bias) const;

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
