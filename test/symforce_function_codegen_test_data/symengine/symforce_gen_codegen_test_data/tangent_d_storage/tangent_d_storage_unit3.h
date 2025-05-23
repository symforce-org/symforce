// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     function/FUNCTION.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>

#include <sym/unit3.h>

namespace sym {

/**
 * Computes the jacobian of the tangent space around an element with respect to the storage space of
 * that element.
 */
template <typename Scalar>
Eigen::Matrix<Scalar, 2, 3> TangentDStorage(const sym::Unit3<Scalar>& a, const Scalar epsilon) {
  // Total ops: 48

  // Input arrays
  const Eigen::Matrix<Scalar, 3, 1>& _a = a.Data();

  // Intermediate terms (14)
  const Scalar _tmp0 = _a[0] - 1;
  const Scalar _tmp1 = std::pow(_a[1], Scalar(2));
  const Scalar _tmp2 =
      std::max<Scalar>(0, -(((std::pow(_a[2], Scalar(2)) + _tmp1 -
                              10 * epsilon * std::copysign(Scalar(1.0), _a[0])) > 0) -
                            ((std::pow(_a[2], Scalar(2)) + _tmp1 -
                              10 * epsilon * std::copysign(Scalar(1.0), _a[0])) < 0)));
  const Scalar _tmp3 = 1 - _tmp2;
  const Scalar _tmp4 = _a[2] + epsilon * std::copysign(Scalar(1.0), _a[2]);
  const Scalar _tmp5 = std::pow(_tmp4, Scalar(2));
  const Scalar _tmp6 = _tmp1 + _tmp5;
  const Scalar _tmp7 = 2 / (std::pow(_tmp0, Scalar(2)) + _tmp6);
  const Scalar _tmp8 = _tmp3 * _tmp7;
  const Scalar _tmp9 = _tmp0 * _tmp8;
  const Scalar _tmp10 = 2 / _tmp6;
  const Scalar _tmp11 = _a[1] * _tmp4;
  const Scalar _tmp12 = _tmp10 * _tmp11 * _tmp2;
  const Scalar _tmp13 = _tmp11 * _tmp8;

  // Output terms (1)
  Eigen::Matrix<Scalar, 2, 3> _res;

  _res(0, 0) = _a[1] * _tmp9;
  _res(1, 0) = -_tmp4 * _tmp9;
  _res(0, 1) = -_tmp2 * (-_tmp1 * _tmp10 + 1) - _tmp3 * (-_tmp1 * _tmp7 + 1);
  _res(1, 1) = -_tmp12 - _tmp13;
  _res(0, 2) = _tmp12 + _tmp13;
  _res(1, 2) = _tmp2 * (-_tmp10 * _tmp5 + 1) + _tmp3 * (-_tmp5 * _tmp7 + 1);

  return _res;
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace sym
