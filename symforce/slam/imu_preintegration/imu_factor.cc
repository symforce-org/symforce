/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./imu_factor.h"

#include <Eigen/Cholesky>

#include <sym/factors/internal/internal_imu_factor.h>

#include "preintegrated_imu_measurements.h"

namespace sym {

template <typename Scalar>
ImuFactor<Scalar>::ImuFactor(const ImuPreintegrator<Scalar>& preintegrator)
    : preintegrated_measurements_{preintegrator.PreintegratedMeasurements()},
      // NOTE(brad, chao): llt then inverse is 2x faster than inverse then llt
      sqrt_info_{preintegrator.Covariance().llt().matrixL().solve(
          Eigen::Matrix<Scalar, 9, 9>::Identity())} {}

template <typename Scalar>
sym::Factor<Scalar> ImuFactor<Scalar>::Factor(const std::vector<Key>& keys_to_func) const {
  const auto begin = keys_to_func.begin();
  // NOTE(brad): *this is copied. Keys to optimize happen to be first 6 keys to func
  return sym::Factor<Scalar>::Hessian(*this, keys_to_func, std::vector<Key>(begin, begin + 6));
}

template <typename Scalar>
void ImuFactor<Scalar>::operator()(
    const sym::Pose3<Scalar>& pose_i, const Eigen::Matrix<Scalar, 3, 1>& vel_i,
    const sym::Pose3<Scalar>& pose_j, const Eigen::Matrix<Scalar, 3, 1>& vel_j,
    const Eigen::Matrix<Scalar, 3, 1>& accel_bias_i, const Eigen::Matrix<Scalar, 3, 1>& gyro_bias_i,
    const Eigen::Matrix<Scalar, 3, 1>& gravity, const Scalar epsilon,
    Eigen::Matrix<Scalar, 9, 1>* const res, Eigen::Matrix<Scalar, 9, 24>* const jacobian,
    Eigen::Matrix<Scalar, 24, 24>* const hessian, Eigen::Matrix<Scalar, 24, 1>* const rhs) const {
  InternalImuFactor(
      pose_i, vel_i, pose_j, vel_j, accel_bias_i, gyro_bias_i, preintegrated_measurements_.DR,
      preintegrated_measurements_.Dv, preintegrated_measurements_.Dp, sqrt_info_,
      preintegrated_measurements_.DR_D_gyro_bias, preintegrated_measurements_.Dv_D_accel_bias,
      preintegrated_measurements_.Dv_D_gyro_bias, preintegrated_measurements_.Dp_D_accel_bias,
      preintegrated_measurements_.Dp_D_gyro_bias, preintegrated_measurements_.accel_bias,
      preintegrated_measurements_.gyro_bias, gravity, preintegrated_measurements_.integrated_dt,
      epsilon,
      // outputs
      res, jacobian, hessian, rhs);
}

}  // namespace sym

template class sym::ImuFactor<double>;
template class sym::ImuFactor<float>;
