// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     geo_package/CLASS.cc.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#include <sym/rot3.h>

namespace sym {

// Print implementations
std::ostream& operator<<(std::ostream& os, const Rot3d& a) {
  const Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n", "[", "]");
  os << "<Rot3d " << a.Data().transpose().format(fmt) << ">";
  return os;
}
std::ostream& operator<<(std::ostream& os, const Rot3f& a) {
  const Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n", "[", "]");
  os << "<Rot3f " << a.Data().transpose().format(fmt) << ">";
  return os;
}

}  // namespace sym

// --------------------------------------------------------------------------
// Custom generated methods
// --------------------------------------------------------------------------

template <typename Scalar>
const Eigen::Matrix<Scalar, 3, 1> sym::Rot3<Scalar>::ComposeWithPoint(
    const Eigen::Matrix<Scalar, 3, 1>& right) const {
  // Total ops: 43

  // Input arrays
  const Eigen::Matrix<Scalar, 4, 1>& _self = Data();

  // Intermediate terms (11)
  const Scalar _tmp0 = 2 * _self[0];
  const Scalar _tmp1 = _self[1] * _tmp0;
  const Scalar _tmp2 = 2 * _self[2];
  const Scalar _tmp3 = _self[3] * _tmp2;
  const Scalar _tmp4 = 2 * _self[1] * _self[3];
  const Scalar _tmp5 = _self[2] * _tmp0;
  const Scalar _tmp6 = -2 * std::pow(_self[1], Scalar(2));
  const Scalar _tmp7 = 1 - 2 * std::pow(_self[2], Scalar(2));
  const Scalar _tmp8 = _self[3] * _tmp0;
  const Scalar _tmp9 = _self[1] * _tmp2;
  const Scalar _tmp10 = -2 * std::pow(_self[0], Scalar(2));

  // Output terms (1)
  Eigen::Matrix<Scalar, 3, 1> _res;

  _res(0, 0) =
      right(0, 0) * (_tmp6 + _tmp7) + right(1, 0) * (_tmp1 - _tmp3) + right(2, 0) * (_tmp4 + _tmp5);
  _res(1, 0) = right(0, 0) * (_tmp1 + _tmp3) + right(1, 0) * (_tmp10 + _tmp7) +
               right(2, 0) * (-_tmp8 + _tmp9);
  _res(2, 0) = right(0, 0) * (-_tmp4 + _tmp5) + right(1, 0) * (_tmp8 + _tmp9) +
               right(2, 0) * (_tmp10 + _tmp6 + 1);

  return _res;
}

template <typename Scalar>
const Scalar sym::Rot3<Scalar>::ToTangentNorm(const Scalar epsilon) const {
  // Total ops: 5

  // Input arrays
  const Eigen::Matrix<Scalar, 4, 1>& _self = Data();

  // Intermediate terms (0)

  // Output terms (1)
  Scalar _res;

  _res = 2 * std::acos(std::min<Scalar>(std::fabs(_self[3]), 1 - epsilon));

  return _res;
}

template <typename Scalar>
const Eigen::Matrix<Scalar, 3, 3> sym::Rot3<Scalar>::ToRotationMatrix() const {
  // Total ops: 28

  // Input arrays
  const Eigen::Matrix<Scalar, 4, 1>& _self = Data();

  // Intermediate terms (11)
  const Scalar _tmp0 = -2 * std::pow(_self[1], Scalar(2));
  const Scalar _tmp1 = 1 - 2 * std::pow(_self[2], Scalar(2));
  const Scalar _tmp2 = 2 * _self[0];
  const Scalar _tmp3 = _self[1] * _tmp2;
  const Scalar _tmp4 = 2 * _self[2];
  const Scalar _tmp5 = _self[3] * _tmp4;
  const Scalar _tmp6 = 2 * _self[1] * _self[3];
  const Scalar _tmp7 = _self[2] * _tmp2;
  const Scalar _tmp8 = -2 * std::pow(_self[0], Scalar(2));
  const Scalar _tmp9 = _self[3] * _tmp2;
  const Scalar _tmp10 = _self[1] * _tmp4;

  // Output terms (1)
  Eigen::Matrix<Scalar, 3, 3> _res;

  _res(0, 0) = _tmp0 + _tmp1;
  _res(1, 0) = _tmp3 + _tmp5;
  _res(2, 0) = -_tmp6 + _tmp7;
  _res(0, 1) = _tmp3 - _tmp5;
  _res(1, 1) = _tmp1 + _tmp8;
  _res(2, 1) = _tmp10 + _tmp9;
  _res(0, 2) = _tmp6 + _tmp7;
  _res(1, 2) = _tmp10 - _tmp9;
  _res(2, 2) = _tmp0 + _tmp8 + 1;

  return _res;
}

template <typename Scalar>
const sym::Rot3<Scalar> sym::Rot3<Scalar>::RandomFromUniformSamples(const Scalar u1,
                                                                    const Scalar u2,
                                                                    const Scalar u3) {
  // Total ops: 14

  // Input arrays

  // Intermediate terms (5)
  const Scalar _tmp0 = std::sqrt(Scalar(1 - u1));
  const Scalar _tmp1 = 2 * Scalar(M_PI);
  const Scalar _tmp2 = _tmp1 * u2;
  const Scalar _tmp3 = std::sqrt(u1);
  const Scalar _tmp4 = _tmp1 * u3;

  // Output terms (1)
  Eigen::Matrix<Scalar, 4, 1> _res;

  _res[0] = _tmp0 * std::sin(_tmp2);
  _res[1] = _tmp0 * std::cos(_tmp2);
  _res[2] = _tmp3 * std::sin(_tmp4);
  _res[3] = _tmp3 * std::cos(_tmp4);

  return sym::Rot3<Scalar>(_res);
}

template <typename Scalar>
const Eigen::Matrix<Scalar, 3, 1> sym::Rot3<Scalar>::ToYawPitchRoll() const {
  // Total ops: 27

  // Input arrays
  const Eigen::Matrix<Scalar, 4, 1>& _self = Data();

  // Intermediate terms (7)
  const Scalar _tmp0 = 2 * _self[0];
  const Scalar _tmp1 = 2 * _self[2];
  const Scalar _tmp2 = std::pow(_self[2], Scalar(2));
  const Scalar _tmp3 = std::pow(_self[0], Scalar(2));
  const Scalar _tmp4 = -std::pow(_self[1], Scalar(2)) + std::pow(_self[3], Scalar(2));
  const Scalar _tmp5 = -_tmp2 + _tmp3 + _tmp4;
  const Scalar _tmp6 = _tmp2 - _tmp3 + _tmp4;

  // Output terms (1)
  Eigen::Matrix<Scalar, 3, 1> _res;

  _res(0, 0) = std::atan2(_self[1] * _tmp0 + _self[3] * _tmp1, _tmp5);
  _res(1, 0) = -std::asin(
      std::max<Scalar>(-1, std::min<Scalar>(1, -2 * _self[1] * _self[3] + _self[2] * _tmp0)));
  _res(2, 0) = std::atan2(_self[1] * _tmp1 + _self[3] * _tmp0, _tmp6);

  return _res;
}

template <typename Scalar>
const sym::Rot3<Scalar> sym::Rot3<Scalar>::FromYawPitchRoll(const Scalar yaw, const Scalar pitch,
                                                            const Scalar roll) {
  // Total ops: 25

  // Input arrays

  // Intermediate terms (13)
  const Scalar _tmp0 = (Scalar(1) / Scalar(2)) * pitch;
  const Scalar _tmp1 = std::sin(_tmp0);
  const Scalar _tmp2 = (Scalar(1) / Scalar(2)) * yaw;
  const Scalar _tmp3 = std::sin(_tmp2);
  const Scalar _tmp4 = (Scalar(1) / Scalar(2)) * roll;
  const Scalar _tmp5 = std::cos(_tmp4);
  const Scalar _tmp6 = _tmp3 * _tmp5;
  const Scalar _tmp7 = std::cos(_tmp0);
  const Scalar _tmp8 = std::sin(_tmp4);
  const Scalar _tmp9 = std::cos(_tmp2);
  const Scalar _tmp10 = _tmp8 * _tmp9;
  const Scalar _tmp11 = _tmp3 * _tmp8;
  const Scalar _tmp12 = _tmp5 * _tmp9;

  // Output terms (1)
  Eigen::Matrix<Scalar, 4, 1> _res;

  _res[0] = -_tmp1 * _tmp6 + _tmp10 * _tmp7;
  _res[1] = _tmp1 * _tmp12 + _tmp11 * _tmp7;
  _res[2] = -_tmp1 * _tmp10 + _tmp6 * _tmp7;
  _res[3] = _tmp1 * _tmp11 + _tmp12 * _tmp7;

  return sym::Rot3<Scalar>(_res);
}

template <typename Scalar>
const sym::Rot3<Scalar> sym::Rot3<Scalar>::FromYaw(const Scalar yaw) {
  // Total ops: 5

  // Input arrays

  // Intermediate terms (1)
  const Scalar _tmp0 = (Scalar(1) / Scalar(2)) * yaw;

  // Output terms (1)
  Eigen::Matrix<Scalar, 4, 1> _res;

  _res[0] = 0;
  _res[1] = 0;
  _res[2] = Scalar(1.0) * std::sin(_tmp0);
  _res[3] = Scalar(1.0) * std::cos(_tmp0);

  return sym::Rot3<Scalar>(_res);
}

template <typename Scalar>
const sym::Rot3<Scalar> sym::Rot3<Scalar>::FromPitch(const Scalar pitch) {
  // Total ops: 5

  // Input arrays

  // Intermediate terms (1)
  const Scalar _tmp0 = (Scalar(1) / Scalar(2)) * pitch;

  // Output terms (1)
  Eigen::Matrix<Scalar, 4, 1> _res;

  _res[0] = 0;
  _res[1] = Scalar(1.0) * std::sin(_tmp0);
  _res[2] = 0;
  _res[3] = Scalar(1.0) * std::cos(_tmp0);

  return sym::Rot3<Scalar>(_res);
}

template <typename Scalar>
const sym::Rot3<Scalar> sym::Rot3<Scalar>::FromRoll(const Scalar roll) {
  // Total ops: 5

  // Input arrays

  // Intermediate terms (1)
  const Scalar _tmp0 = (Scalar(1) / Scalar(2)) * roll;

  // Output terms (1)
  Eigen::Matrix<Scalar, 4, 1> _res;

  _res[0] = Scalar(1.0) * std::sin(_tmp0);
  _res[1] = 0;
  _res[2] = 0;
  _res[3] = Scalar(1.0) * std::cos(_tmp0);

  return sym::Rot3<Scalar>(_res);
}

template <typename Scalar>
const sym::Rot3<Scalar> sym::Rot3<Scalar>::FromYawPitchRoll(
    const Eigen::Matrix<Scalar, 3, 1>& ypr) {
  // Total ops: 25

  // Input arrays

  // Intermediate terms (13)
  const Scalar _tmp0 = (Scalar(1) / Scalar(2)) * ypr(2, 0);
  const Scalar _tmp1 = std::sin(_tmp0);
  const Scalar _tmp2 = (Scalar(1) / Scalar(2)) * ypr(1, 0);
  const Scalar _tmp3 = std::cos(_tmp2);
  const Scalar _tmp4 = (Scalar(1) / Scalar(2)) * ypr(0, 0);
  const Scalar _tmp5 = std::cos(_tmp4);
  const Scalar _tmp6 = _tmp3 * _tmp5;
  const Scalar _tmp7 = std::cos(_tmp0);
  const Scalar _tmp8 = std::sin(_tmp4);
  const Scalar _tmp9 = std::sin(_tmp2);
  const Scalar _tmp10 = _tmp8 * _tmp9;
  const Scalar _tmp11 = _tmp3 * _tmp8;
  const Scalar _tmp12 = _tmp5 * _tmp9;

  // Output terms (1)
  Eigen::Matrix<Scalar, 4, 1> _res;

  _res[0] = _tmp1 * _tmp6 - _tmp10 * _tmp7;
  _res[1] = _tmp1 * _tmp11 + _tmp12 * _tmp7;
  _res[2] = -_tmp1 * _tmp12 + _tmp11 * _tmp7;
  _res[3] = _tmp1 * _tmp10 + _tmp6 * _tmp7;

  return sym::Rot3<Scalar>(_res);
}

template <typename Scalar>
const sym::Rot3<Scalar> sym::Rot3<Scalar>::FromTwoUnitVectors(const Eigen::Matrix<Scalar, 3, 1>& a,
                                                              const Eigen::Matrix<Scalar, 3, 1>& b,
                                                              const Scalar epsilon) {
  // Total ops: 44

  // Input arrays

  // Intermediate terms (7)
  const Scalar _tmp0 = a(0, 0) * b(0, 0) + a(1, 0) * b(1, 0) + a(2, 0) * b(2, 0);
  const Scalar _tmp1 = std::sqrt(Scalar(2 * _tmp0 + epsilon + 2));
  const Scalar _tmp2 =
      (((-epsilon + std::fabs(_tmp0 + 1)) > 0) - ((-epsilon + std::fabs(_tmp0 + 1)) < 0)) + 1;
  const Scalar _tmp3 = (Scalar(1) / Scalar(2)) * _tmp2;
  const Scalar _tmp4 = _tmp3 / _tmp1;
  const Scalar _tmp5 =
      Scalar(1) / Scalar(2) - Scalar(1) / Scalar(2) *
                                  (((std::pow(a(1, 0), Scalar(2)) + std::pow(a(2, 0), Scalar(2)) -
                                     std::pow(epsilon, Scalar(2))) > 0) -
                                   ((std::pow(a(1, 0), Scalar(2)) + std::pow(a(2, 0), Scalar(2)) -
                                     std::pow(epsilon, Scalar(2))) < 0));
  const Scalar _tmp6 = 1 - _tmp3;

  // Output terms (1)
  Eigen::Matrix<Scalar, 4, 1> _res;

  _res[0] = _tmp4 * (a(1, 0) * b(2, 0) - a(2, 0) * b(1, 0)) + _tmp6 * (1 - _tmp5);
  _res[1] = _tmp4 * (-a(0, 0) * b(2, 0) + a(2, 0) * b(0, 0)) + _tmp5 * _tmp6;
  _res[2] = _tmp4 * (a(0, 0) * b(1, 0) - a(1, 0) * b(0, 0));
  _res[3] = (Scalar(1) / Scalar(4)) * _tmp1 * _tmp2;

  return sym::Rot3<Scalar>(_res);
}

// Explicit instantiation
template class sym::Rot3<double>;
template class sym::Rot3<float>;
