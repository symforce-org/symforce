// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     function/FUNCTION.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>

#include <sym/pose2.h>

namespace sym {

/**
 * Residual from a relative bearing measurement of a 2D pose to a landmark.
 *     jacobian: (1x3) jacobian of res wrt arg pose (3)
 *     hessian: (3x3) Gauss-Newton hessian for arg pose (3)
 *     rhs: (3x1) Gauss-Newton rhs for arg pose (3)
 */
template <typename Scalar>
void BearingFactor(const sym::Pose2<Scalar>& pose, const Eigen::Matrix<Scalar, 2, 1>& landmark,
                   const Scalar angle, const Scalar epsilon,
                   Eigen::Matrix<Scalar, 1, 1>* const res = nullptr,
                   Eigen::Matrix<Scalar, 1, 3>* const jacobian = nullptr,
                   Eigen::Matrix<Scalar, 3, 3>* const hessian = nullptr,
                   Eigen::Matrix<Scalar, 3, 1>* const rhs = nullptr) {
  // Total ops: 64

  // Input arrays
  const Eigen::Matrix<Scalar, 4, 1>& _pose = pose.Data();

  // Intermediate terms (24)
  const Scalar _tmp0 = _pose[1] * _pose[2];
  const Scalar _tmp1 = _pose[0] * _pose[3];
  const Scalar _tmp2 = _pose[0] * landmark(1, 0) - _pose[1] * landmark(0, 0);
  const Scalar _tmp3 = _tmp0 - _tmp1 + _tmp2;
  const Scalar _tmp4 = _pose[0] * _pose[2] + _pose[1] * _pose[3];
  const Scalar _tmp5 = _pose[1] * landmark(1, 0);
  const Scalar _tmp6 = _pose[0] * landmark(0, 0);
  const Scalar _tmp7 = -_tmp4 + _tmp5 + _tmp6;
  const Scalar _tmp8 = _tmp7 + std::copysign(epsilon, _tmp7);
  const Scalar _tmp9 = -angle + std::atan2(_tmp3, _tmp8);
  const Scalar _tmp10 =
      _tmp9 - 2 * Scalar(M_PI) *
                  std::floor((Scalar(1) / Scalar(2)) * (_tmp9 + Scalar(M_PI)) / Scalar(M_PI));
  const Scalar _tmp11 = Scalar(1.0) / (_tmp8);
  const Scalar _tmp12 = std::pow(_tmp8, Scalar(2));
  const Scalar _tmp13 = _tmp3 / _tmp12;
  const Scalar _tmp14 = _tmp11 * (_tmp4 - _tmp5 - _tmp6) - _tmp13 * (_tmp0 - _tmp1 + _tmp2);
  const Scalar _tmp15 = _tmp12 + std::pow(_tmp3, Scalar(2));
  const Scalar _tmp16 = _tmp12 / _tmp15;
  const Scalar _tmp17 = _tmp14 * _tmp16;
  const Scalar _tmp18 = _pose[0] * _tmp13 + _pose[1] * _tmp11;
  const Scalar _tmp19 = _tmp16 * _tmp18;
  const Scalar _tmp20 = -_pose[0] * _tmp11 + _pose[1] * _tmp13;
  const Scalar _tmp21 = _tmp16 * _tmp20;
  const Scalar _tmp22 = std::pow(_tmp8, Scalar(4)) / std::pow(_tmp15, Scalar(2));
  const Scalar _tmp23 = _tmp14 * _tmp22;

  // Output terms (4)
  if (res != nullptr) {
    Eigen::Matrix<Scalar, 1, 1>& _res = (*res);

    _res(0, 0) = _tmp10;
  }

  if (jacobian != nullptr) {
    Eigen::Matrix<Scalar, 1, 3>& _jacobian = (*jacobian);

    _jacobian(0, 0) = _tmp17;
    _jacobian(0, 1) = _tmp19;
    _jacobian(0, 2) = _tmp21;
  }

  if (hessian != nullptr) {
    Eigen::Matrix<Scalar, 3, 3>& _hessian = (*hessian);

    _hessian(0, 0) = std::pow(_tmp14, Scalar(2)) * _tmp22;
    _hessian(1, 0) = _tmp18 * _tmp23;
    _hessian(2, 0) = _tmp20 * _tmp23;
    _hessian(0, 1) = 0;
    _hessian(1, 1) = std::pow(_tmp18, Scalar(2)) * _tmp22;
    _hessian(2, 1) = _tmp18 * _tmp20 * _tmp22;
    _hessian(0, 2) = 0;
    _hessian(1, 2) = 0;
    _hessian(2, 2) = std::pow(_tmp20, Scalar(2)) * _tmp22;
  }

  if (rhs != nullptr) {
    Eigen::Matrix<Scalar, 3, 1>& _rhs = (*rhs);

    _rhs(0, 0) = _tmp10 * _tmp17;
    _rhs(1, 0) = _tmp10 * _tmp19;
    _rhs(2, 0) = _tmp10 * _tmp21;
  }
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace sym
