/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <Eigen/Core>

#include <sym/util/epsilon.h>

#include "./preintegrated_imu_measurements.h"

namespace sym {

/**
 * Class to on-manifold preintegrate IMU measurements for usage in a SymForce optimization
 * problem.
 *
 * For usage, see the doc-string of ImuFactor in imu_factor.h
 */
template <typename Scalar>
class ImuPreintegrator {
 public:
  using Vector3 = typename PreintegratedImuMeasurements<Scalar>::Vector3;
  using Matrix33 = typename PreintegratedImuMeasurements<Scalar>::Matrix33;
  using Matrix99 = Eigen::Matrix<Scalar, 9, 9>;

 private:
  PreintegratedImuMeasurements<Scalar> preintegrated_measurements_;
  Matrix99 covariance_;  // covariance of [DR, Dv, Dp] in local coordinates of mean

 public:
  /**
   * Initialize with given accel_bias and gyro_bias
   */
  ImuPreintegrator(const Vector3& accel_bias, const Vector3& gyro_bias);

  /**
   * Integrate a new measurement.
   *
   *   @param measured_accel the next accelerometer measurement after the last integrated
   * measurement
   *   @param measured_gyro the next gyroscope measurement after the last integrated measurement
   *   @param accel_cov the covariance of the accelerometer measurement (represented by its diagonal
   *     entries) `[(meters / time^2)^2 * time]`
   *   @param gyro_cov the covariance of the gyroscope measurement (represented by its diagonal
   *     entries) `[(radians / time)^2 * time]`
   *   @param dt the time span over which these measurements were made
   *
   * Postcondition:
   *
   *   If the orientation, velocity, and position were R0, v0, and p0 at the start of the first
   *   integrated IMU measurements, and Rf, vf, and pf are the corresponding values at the end of
   *   the measurement described by the arguments, DT is the total time covered by the integrated
   *   measurements, and g is the vector of acceleration due to gravity, then
   *   - pim.DR -> R0.inverse() * Rf
   *   - pim.Dv -> R0.inverse() * (vf - v0 - g * DT)
   *   - pim.Dp -> R0.inverse() * (pf - p0 - v0 * DT - 0.5 * g * DT^2)
   *   - pim.DX_D_accel_bias -> the derivative of DX wrt the accel bias linearized at pim.accel_bias
   *   - pim.DX_D_gyro_bias -> the derivative of DX wrt the gyro bias linearized at pim.gyro_bias
   *   - pim.accel_bias -> unchanged
   *   - pim.gyro_bias -> unchanged
   *   - pim.integrated_dt -> DT
   *   - covariance -> the covariance [DR, Dv, Dp] in local coordinates of mean
   */
  void IntegrateMeasurement(const Vector3& measured_accel, const Vector3& measured_gyro,
                            const Vector3& accel_cov, const Vector3& gyro_cov, Scalar dt,
                            Scalar epsilon = kDefaultEpsilon<Scalar>);

  const PreintegratedImuMeasurements<Scalar>& PreintegratedMeasurements() const;

  const Matrix99& Covariance() const;
};

using ImuPreintegratord = ImuPreintegrator<double>;
using ImuPreintegratorf = ImuPreintegrator<float>;

}  // namespace sym

extern template class sym::ImuPreintegrator<double>;
extern template class sym::ImuPreintegrator<float>;
