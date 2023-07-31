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
ImuFactor<Scalar>::ImuFactor(const Preintegrator& preintegrator)
    : ImuFactor{preintegrator.PreintegratedMeasurements(),
                preintegrator.Covariance().llt().matrixL().solve(SqrtInformation::Identity())} {}

template <typename Scalar>
ImuFactor<Scalar>::ImuFactor(const Measurement& measurement,
                             const SqrtInformation& sqrt_information)
    : measurement_{measurement}, sqrt_information_{sqrt_information} {}

template <typename Scalar>
sym::Factor<Scalar> ImuFactor<Scalar>::Factor(const std::vector<Key>& keys_to_func) const {
  const auto begin = keys_to_func.begin();
  // NOTE(brad): *this is copied. Keys to optimize happen to be first 6 keys to func
  return sym::Factor<Scalar>::Hessian(*this, keys_to_func, std::vector<Key>(begin, begin + 6));
}

template <typename Scalar>
void ImuFactor<Scalar>::operator()(const Pose3& pose_i, const Vector3& vel_i, const Pose3& pose_j,
                                   const Vector3& vel_j, const Vector3& accel_bias_i,
                                   const Vector3& gyro_bias_i, const Vector3& gravity,
                                   const Scalar epsilon,
                                   Eigen::Matrix<Scalar, 9, 1>* const residual,
                                   Eigen::Matrix<Scalar, 9, 24>* const jacobian,
                                   Eigen::Matrix<Scalar, 24, 24>* const hessian,
                                   Eigen::Matrix<Scalar, 24, 1>* const rhs) const {
  InternalImuFactor(pose_i, vel_i, pose_j, vel_j, accel_bias_i, gyro_bias_i, measurement_.DR,
                    measurement_.Dv, measurement_.Dp, sqrt_information_,
                    measurement_.DR_D_gyro_bias, measurement_.Dv_D_accel_bias,
                    measurement_.Dv_D_gyro_bias, measurement_.Dp_D_accel_bias,
                    measurement_.Dp_D_gyro_bias, measurement_.accel_bias, measurement_.gyro_bias,
                    gravity, measurement_.integrated_dt, epsilon, residual, jacobian, hessian, rhs);
}

}  // namespace sym

template class sym::ImuFactor<double>;
template class sym::ImuFactor<float>;
