/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./imu_factor.h"

#include <Eigen/Cholesky>

#include <sym/factors/internal/internal_imu_factor.h>
#include <sym/factors/internal/internal_imu_unit_gravity_factor.h>
#include <sym/factors/internal/internal_imu_with_gravity_factor.h>

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
  const auto& delta = measurement_.delta;
  InternalImuFactor(pose_i, vel_i, pose_j, vel_j, accel_bias_i, gyro_bias_i, delta.DR, delta.Dv,
                    delta.Dp, sqrt_information_, measurement_.DR_D_gyro_bias,
                    measurement_.Dv_D_accel_bias, measurement_.Dv_D_gyro_bias,
                    measurement_.Dp_D_accel_bias, measurement_.Dp_D_gyro_bias,
                    measurement_.accel_bias, measurement_.gyro_bias, gravity, delta.Dt, epsilon,
                    residual, jacobian, hessian, rhs);
}

template <typename Scalar>
ImuWithGravityFactor<Scalar>::ImuWithGravityFactor(const Preintegrator& preintegrator)
    : ImuWithGravityFactor{
          preintegrator.PreintegratedMeasurements(),
          preintegrator.Covariance().llt().matrixL().solve(SqrtInformation::Identity())} {}

template <typename Scalar>
ImuWithGravityFactor<Scalar>::ImuWithGravityFactor(const Measurement& measurement,
                                                   const SqrtInformation& sqrt_information)
    : measurement_{measurement}, sqrt_information_{sqrt_information} {}

template <typename Scalar>
sym::Factor<Scalar> ImuWithGravityFactor<Scalar>::Factor(
    const std::vector<Key>& keys_to_func) const {
  const auto begin = keys_to_func.begin();
  // NOTE(brad): *this is copied. Keys to optimize happen to be first 7 keys to func
  return sym::Factor<Scalar>::Hessian(*this, keys_to_func, std::vector<Key>(begin, begin + 7));
}

template <typename Scalar>
void ImuWithGravityFactor<Scalar>::operator()(
    const Pose3& pose_i, const Vector3& vel_i, const Pose3& pose_j, const Vector3& vel_j,
    const Vector3& accel_bias_i, const Vector3& gyro_bias_i, const Vector3& gravity,
    const Scalar epsilon, Eigen::Matrix<Scalar, 9, 1>* const residual,
    Eigen::Matrix<Scalar, 9, 27>* const jacobian, Eigen::Matrix<Scalar, 27, 27>* const hessian,
    Eigen::Matrix<Scalar, 27, 1>* const rhs) const {
  const auto& delta = measurement_.delta;
  InternalImuWithGravityFactor(pose_i, vel_i, pose_j, vel_j, accel_bias_i, gyro_bias_i, delta.DR,
                               delta.Dv, delta.Dp, sqrt_information_, measurement_.DR_D_gyro_bias,
                               measurement_.Dv_D_accel_bias, measurement_.Dv_D_gyro_bias,
                               measurement_.Dp_D_accel_bias, measurement_.Dp_D_gyro_bias,
                               measurement_.accel_bias, measurement_.gyro_bias, gravity, delta.Dt,
                               epsilon, residual, jacobian, hessian, rhs);
}

template <typename Scalar>
ImuWithGravityDirectionFactor<Scalar>::ImuWithGravityDirectionFactor(
    const Preintegrator& preintegrator)
    : ImuWithGravityDirectionFactor{
          preintegrator.PreintegratedMeasurements(),
          preintegrator.Covariance().llt().matrixL().solve(SqrtInformation::Identity())} {}

template <typename Scalar>
ImuWithGravityDirectionFactor<Scalar>::ImuWithGravityDirectionFactor(
    const Measurement& measurement, const SqrtInformation& sqrt_information)
    : measurement_{measurement}, sqrt_information_{sqrt_information} {}

template <typename Scalar>
sym::Factor<Scalar> ImuWithGravityDirectionFactor<Scalar>::Factor(
    const std::vector<Key>& keys_to_func) const {
  const auto begin = keys_to_func.begin();
  // NOTE(brad): *this is copied. Keys to optimize happen to be first 7 keys to func
  return sym::Factor<Scalar>::Hessian(*this, keys_to_func, std::vector<Key>(begin, begin + 7));
}

template <typename Scalar>
void ImuWithGravityDirectionFactor<Scalar>::operator()(
    const Pose3& pose_i, const Vector3& vel_i, const Pose3& pose_j, const Vector3& vel_j,
    const Vector3& accel_bias_i, const Vector3& gyro_bias_i, const Unit3<Scalar>& gravity_direction,
    const Scalar gravity_norm, const Scalar epsilon, Eigen::Matrix<Scalar, 9, 1>* const residual,
    Eigen::Matrix<Scalar, 9, 26>* const jacobian, Eigen::Matrix<Scalar, 26, 26>* const hessian,
    Eigen::Matrix<Scalar, 26, 1>* const rhs) const {
  const auto& delta = measurement_.delta;
  InternalImuUnitGravityFactor(pose_i, vel_i, pose_j, vel_j, accel_bias_i, gyro_bias_i, delta.DR,
                               delta.Dv, delta.Dp, sqrt_information_, measurement_.DR_D_gyro_bias,
                               measurement_.Dv_D_accel_bias, measurement_.Dv_D_gyro_bias,
                               measurement_.Dp_D_accel_bias, measurement_.Dp_D_gyro_bias,
                               measurement_.accel_bias, measurement_.gyro_bias, gravity_direction,
                               gravity_norm, delta.Dt, epsilon, residual, jacobian, hessian, rhs);
}

}  // namespace sym

template class sym::ImuFactor<double>;
template class sym::ImuFactor<float>;
template class sym::ImuWithGravityFactor<double>;
template class sym::ImuWithGravityFactor<float>;
template class sym::ImuWithGravityDirectionFactor<double>;
template class sym::ImuWithGravityDirectionFactor<float>;
