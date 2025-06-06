// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     geo_package/CLASS.cc.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#include <sym/unit3.h>

namespace sym {

// Print implementations
std::ostream& operator<<(std::ostream& os, const Unit3d& a) {
  const Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n", "[", "]");
  os << "<Unit3d " << a.Data().transpose().format(fmt) << ">";
  return os;
}
std::ostream& operator<<(std::ostream& os, const Unit3f& a) {
  const Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n", "[", "]");
  os << "<Unit3f " << a.Data().transpose().format(fmt) << ">";
  return os;
}

}  // namespace sym

// --------------------------------------------------------------------------
// Custom generated methods
// --------------------------------------------------------------------------

template <typename Scalar>
const Eigen::Matrix<Scalar, 3, 2> sym::Unit3<Scalar>::Basis(const Scalar epsilon) const {
  // Total ops: 50

  // Input arrays
  const Eigen::Matrix<Scalar, 3, 1>& _self = Data();

  // Intermediate terms (12)
  const Scalar _tmp0 = std::pow(_self[1], Scalar(2));
  const Scalar _tmp1 =
      std::max<Scalar>(0, -(((std::pow(_self[2], Scalar(2)) + _tmp0 -
                              10 * epsilon * std::copysign(Scalar(1.0), _self[0])) > 0) -
                            ((std::pow(_self[2], Scalar(2)) + _tmp0 -
                              10 * epsilon * std::copysign(Scalar(1.0), _self[0])) < 0)));
  const Scalar _tmp2 = 1 - _tmp1;
  const Scalar _tmp3 = _self[0] - 1;
  const Scalar _tmp4 = _self[2] + epsilon * std::copysign(Scalar(1.0), _self[2]);
  const Scalar _tmp5 = std::pow(_tmp4, Scalar(2));
  const Scalar _tmp6 = _tmp0 + _tmp5;
  const Scalar _tmp7 = 2 / (std::pow(_tmp3, Scalar(2)) + _tmp6);
  const Scalar _tmp8 = 2 / _tmp6;
  const Scalar _tmp9 = _tmp2 * _tmp4 * _tmp7;
  const Scalar _tmp10 = _self[1] * _tmp9;
  const Scalar _tmp11 = _self[1] * _tmp1 * _tmp4 * _tmp8;

  // Output terms (1)
  Eigen::Matrix<Scalar, 3, 2> _res;

  _res(0, 0) = _self[1] * _tmp2 * _tmp3 * _tmp7;
  _res(1, 0) = -_tmp1 * (-_tmp0 * _tmp8 + 1) - _tmp2 * (-_tmp0 * _tmp7 + 1);
  _res(2, 0) = _tmp10 + _tmp11;
  _res(0, 1) = -_tmp3 * _tmp9;
  _res(1, 1) = -_tmp10 - _tmp11;
  _res(2, 1) = _tmp1 * (-_tmp5 * _tmp8 + 1) + _tmp2 * (-_tmp5 * _tmp7 + 1);

  return _res;
}

template <typename Scalar>
const Eigen::Matrix<Scalar, 3, 1> sym::Unit3<Scalar>::ToUnitVector() const {
  // Total ops: 0

  // Input arrays
  const Eigen::Matrix<Scalar, 3, 1>& _self = Data();

  // Intermediate terms (0)

  // Output terms (1)
  Eigen::Matrix<Scalar, 3, 1> _res;

  _res(0, 0) = _self[0];
  _res(1, 0) = _self[1];
  _res(2, 0) = _self[2];

  return _res;
}

template <typename Scalar>
const sym::Unit3<Scalar> sym::Unit3<Scalar>::RandomFromUniformSamples(const Scalar u1,
                                                                      const Scalar u2,
                                                                      const Scalar epsilon) {
  // Total ops: 23

  // Input arrays

  // Intermediate terms (8)
  const Scalar _tmp0 = 2 * Scalar(M_PI) * u1;
  const Scalar _tmp1 = std::cos(_tmp0);
  const Scalar _tmp2 = 2 * u2 - 1;
  const Scalar _tmp3 = std::sqrt(Scalar(1 - std::pow(_tmp2, Scalar(2))));
  const Scalar _tmp4 = std::pow(_tmp3, Scalar(2));
  const Scalar _tmp5 = std::sin(_tmp0);
  const Scalar _tmp6 =
      std::pow(Scalar(std::pow(_tmp1, Scalar(2)) * _tmp4 + std::pow(_tmp2, Scalar(2)) +
                      _tmp4 * std::pow(_tmp5, Scalar(2)) + epsilon),
               Scalar(Scalar(-1) / Scalar(2)));
  const Scalar _tmp7 = _tmp3 * _tmp6;

  // Output terms (1)
  Eigen::Matrix<Scalar, 3, 1> _res;

  _res[0] = _tmp1 * _tmp7;
  _res[1] = _tmp5 * _tmp7;
  _res[2] = _tmp2 * _tmp6;

  return sym::Unit3<Scalar>(_res);
}

// Explicit instantiation
template class sym::Unit3<double>;
template class sym::Unit3<float>;
