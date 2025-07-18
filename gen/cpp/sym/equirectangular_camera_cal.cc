// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     cam_package/CLASS.cc.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#include <sym/equirectangular_camera_cal.h>

// Camera operation implementations
namespace sym {

template <typename Scalar>
Eigen::Matrix<Scalar, 2, 1> EquirectangularCameraCal<Scalar>::FocalLength() const {
  // Total ops: 0

  // Input arrays
  const Eigen::Matrix<Scalar, 4, 1>& _self = Data();

  // Intermediate terms (0)

  // Output terms (1)
  Eigen::Matrix<Scalar, 2, 1> _focal_length;

  _focal_length(0, 0) = _self[0];
  _focal_length(1, 0) = _self[1];

  return _focal_length;
}

template <typename Scalar>
Eigen::Matrix<Scalar, 2, 1> EquirectangularCameraCal<Scalar>::PrincipalPoint() const {
  // Total ops: 0

  // Input arrays
  const Eigen::Matrix<Scalar, 4, 1>& _self = Data();

  // Intermediate terms (0)

  // Output terms (1)
  Eigen::Matrix<Scalar, 2, 1> _principal_point;

  _principal_point(0, 0) = _self[2];
  _principal_point(1, 0) = _self[3];

  return _principal_point;
}

template <typename Scalar>
Eigen::Matrix<Scalar, 2, 1> EquirectangularCameraCal<Scalar>::PixelFromCameraPoint(
    const Eigen::Matrix<Scalar, 3, 1>& point, const Scalar epsilon, Scalar* const is_valid) const {
  // Total ops: 17

  // Input arrays
  const Eigen::Matrix<Scalar, 4, 1>& _self = Data();

  // Intermediate terms (2)
  const Scalar _tmp0 = std::pow(point(0, 0), Scalar(2)) + std::pow(point(2, 0), Scalar(2));
  const Scalar _tmp1 = std::sqrt(Scalar(_tmp0 + epsilon));

  // Output terms (2)
  Eigen::Matrix<Scalar, 2, 1> _pixel;

  _pixel(0, 0) =
      _self[0] * std::atan2(point(0, 0), point(2, 0) + std::copysign(epsilon, point(2, 0))) +
      _self[2];
  _pixel(1, 0) = _self[1] * std::atan2(point(1, 0), _tmp1) + _self[3];

  if (is_valid != nullptr) {
    Scalar& _is_valid = (*is_valid);

    _is_valid = std::max<Scalar>(0, (((_tmp0 + std::pow(point(1, 0), Scalar(2))) > 0) -
                                     ((_tmp0 + std::pow(point(1, 0), Scalar(2))) < 0)));
  }

  return _pixel;
}

template <typename Scalar>
Eigen::Matrix<Scalar, 2, 1> EquirectangularCameraCal<Scalar>::PixelFromCameraPointWithJacobians(
    const Eigen::Matrix<Scalar, 3, 1>& point, const Scalar epsilon, Scalar* const is_valid,
    Eigen::Matrix<Scalar, 2, 4>* const pixel_D_cal,
    Eigen::Matrix<Scalar, 2, 3>* const pixel_D_point) const {
  // Total ops: 33

  // Input arrays
  const Eigen::Matrix<Scalar, 4, 1>& _self = Data();

  // Intermediate terms (11)
  const Scalar _tmp0 = point(2, 0) + std::copysign(epsilon, point(2, 0));
  const Scalar _tmp1 = std::atan2(point(0, 0), _tmp0);
  const Scalar _tmp2 = std::pow(point(0, 0), Scalar(2));
  const Scalar _tmp3 = _tmp2 + std::pow(point(2, 0), Scalar(2));
  const Scalar _tmp4 = std::sqrt(Scalar(_tmp3 + epsilon));
  const Scalar _tmp5 = _tmp4;
  const Scalar _tmp6 = std::atan2(point(1, 0), _tmp5);
  const Scalar _tmp7 = std::pow(point(1, 0), Scalar(2));
  const Scalar _tmp8 = _self[0] / (std::pow(_tmp0, Scalar(2)) + _tmp2);
  const Scalar _tmp9 = _self[1] / (std::pow(_tmp5, Scalar(2)) + _tmp7);
  const Scalar _tmp10 = _tmp9 * point(1, 0) / _tmp4;

  // Output terms (4)
  Eigen::Matrix<Scalar, 2, 1> _pixel;

  _pixel(0, 0) = _self[0] * _tmp1 + _self[2];
  _pixel(1, 0) = _self[1] * _tmp6 + _self[3];

  if (is_valid != nullptr) {
    Scalar& _is_valid = (*is_valid);

    _is_valid = std::max<Scalar>(0, (((_tmp3 + _tmp7) > 0) - ((_tmp3 + _tmp7) < 0)));
  }

  if (pixel_D_cal != nullptr) {
    Eigen::Matrix<Scalar, 2, 4>& _pixel_D_cal = (*pixel_D_cal);

    _pixel_D_cal(0, 0) = _tmp1;
    _pixel_D_cal(1, 0) = 0;
    _pixel_D_cal(0, 1) = 0;
    _pixel_D_cal(1, 1) = _tmp6;
    _pixel_D_cal(0, 2) = 1;
    _pixel_D_cal(1, 2) = 0;
    _pixel_D_cal(0, 3) = 0;
    _pixel_D_cal(1, 3) = 1;
  }

  if (pixel_D_point != nullptr) {
    Eigen::Matrix<Scalar, 2, 3>& _pixel_D_point = (*pixel_D_point);

    _pixel_D_point(0, 0) = _tmp0 * _tmp8;
    _pixel_D_point(1, 0) = -_tmp10 * point(0, 0);
    _pixel_D_point(0, 1) = 0;
    _pixel_D_point(1, 1) = _tmp5 * _tmp9;
    _pixel_D_point(0, 2) = -_tmp8 * point(0, 0);
    _pixel_D_point(1, 2) = -_tmp10 * point(2, 0);
  }

  return _pixel;
}

template <typename Scalar>
Eigen::Matrix<Scalar, 3, 1> EquirectangularCameraCal<Scalar>::CameraRayFromPixel(
    const Eigen::Matrix<Scalar, 2, 1>& pixel, const Scalar epsilon, Scalar* const is_valid) const {
  // Total ops: 19

  // Unused inputs
  (void)epsilon;

  // Input arrays
  const Eigen::Matrix<Scalar, 4, 1>& _self = Data();

  // Intermediate terms (3)
  const Scalar _tmp0 = (-_self[2] + pixel(0, 0)) / _self[0];
  const Scalar _tmp1 = (-_self[3] + pixel(1, 0)) / _self[1];
  const Scalar _tmp2 = std::cos(_tmp1);

  // Output terms (2)
  Eigen::Matrix<Scalar, 3, 1> _camera_ray;

  _camera_ray(0, 0) = _tmp2 * std::sin(_tmp0);
  _camera_ray(1, 0) = std::sin(_tmp1);
  _camera_ray(2, 0) = _tmp2 * std::cos(_tmp0);

  if (is_valid != nullptr) {
    Scalar& _is_valid = (*is_valid);

    _is_valid = std::max<Scalar>(0, std::min<Scalar>((((Scalar(M_PI) - std::fabs(_tmp0)) > 0) -
                                                      ((Scalar(M_PI) - std::fabs(_tmp0)) < 0)),
                                                     (((-std::fabs(_tmp1) + Scalar(M_PI_2)) > 0) -
                                                      ((-std::fabs(_tmp1) + Scalar(M_PI_2)) < 0))));
  }

  return _camera_ray;
}

template <typename Scalar>
Eigen::Matrix<Scalar, 3, 1> EquirectangularCameraCal<Scalar>::CameraRayFromPixelWithJacobians(
    const Eigen::Matrix<Scalar, 2, 1>& pixel, const Scalar epsilon, Scalar* const is_valid,
    Eigen::Matrix<Scalar, 3, 4>* const point_D_cal,
    Eigen::Matrix<Scalar, 3, 2>* const point_D_pixel) const {
  // Total ops: 44

  // Unused inputs
  (void)epsilon;

  // Input arrays
  const Eigen::Matrix<Scalar, 4, 1>& _self = Data();

  // Intermediate terms (21)
  const Scalar _tmp0 = -_self[2] + pixel(0, 0);
  const Scalar _tmp1 = Scalar(1.0) / (_self[0]);
  const Scalar _tmp2 = _tmp0 * _tmp1;
  const Scalar _tmp3 = std::sin(_tmp2);
  const Scalar _tmp4 = -_self[3] + pixel(1, 0);
  const Scalar _tmp5 = Scalar(1.0) / (_self[1]);
  const Scalar _tmp6 = _tmp4 * _tmp5;
  const Scalar _tmp7 = std::cos(_tmp6);
  const Scalar _tmp8 = _tmp3 * _tmp7;
  const Scalar _tmp9 = std::sin(_tmp6);
  const Scalar _tmp10 = std::cos(_tmp2);
  const Scalar _tmp11 = _tmp10 * _tmp7;
  const Scalar _tmp12 = _tmp0 / std::pow(_self[0], Scalar(2));
  const Scalar _tmp13 = _tmp4 / std::pow(_self[1], Scalar(2));
  const Scalar _tmp14 = _tmp13 * _tmp9;
  const Scalar _tmp15 = _tmp1 * _tmp11;
  const Scalar _tmp16 = _tmp1 * _tmp8;
  const Scalar _tmp17 = _tmp5 * _tmp9;
  const Scalar _tmp18 = _tmp17 * _tmp3;
  const Scalar _tmp19 = _tmp5 * _tmp7;
  const Scalar _tmp20 = _tmp10 * _tmp17;

  // Output terms (4)
  Eigen::Matrix<Scalar, 3, 1> _camera_ray;

  _camera_ray(0, 0) = _tmp8;
  _camera_ray(1, 0) = _tmp9;
  _camera_ray(2, 0) = _tmp11;

  if (is_valid != nullptr) {
    Scalar& _is_valid = (*is_valid);

    _is_valid = std::max<Scalar>(0, std::min<Scalar>((((Scalar(M_PI) - std::fabs(_tmp2)) > 0) -
                                                      ((Scalar(M_PI) - std::fabs(_tmp2)) < 0)),
                                                     (((-std::fabs(_tmp6) + Scalar(M_PI_2)) > 0) -
                                                      ((-std::fabs(_tmp6) + Scalar(M_PI_2)) < 0))));
  }

  if (point_D_cal != nullptr) {
    Eigen::Matrix<Scalar, 3, 4>& _point_D_cal = (*point_D_cal);

    _point_D_cal(0, 0) = -_tmp11 * _tmp12;
    _point_D_cal(1, 0) = 0;
    _point_D_cal(2, 0) = _tmp12 * _tmp8;
    _point_D_cal(0, 1) = _tmp14 * _tmp3;
    _point_D_cal(1, 1) = -_tmp13 * _tmp7;
    _point_D_cal(2, 1) = _tmp10 * _tmp14;
    _point_D_cal(0, 2) = -_tmp15;
    _point_D_cal(1, 2) = 0;
    _point_D_cal(2, 2) = _tmp16;
    _point_D_cal(0, 3) = _tmp18;
    _point_D_cal(1, 3) = -_tmp19;
    _point_D_cal(2, 3) = _tmp20;
  }

  if (point_D_pixel != nullptr) {
    Eigen::Matrix<Scalar, 3, 2>& _point_D_pixel = (*point_D_pixel);

    _point_D_pixel(0, 0) = _tmp15;
    _point_D_pixel(1, 0) = 0;
    _point_D_pixel(2, 0) = -_tmp16;
    _point_D_pixel(0, 1) = -_tmp18;
    _point_D_pixel(1, 1) = _tmp19;
    _point_D_pixel(2, 1) = -_tmp20;
  }

  return _camera_ray;
}

// Print implementations
std::ostream& operator<<(std::ostream& os, const EquirectangularCameraCald& a) {
  const Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n", "[", "]");
  os << "<EquirectangularCameraCald " << a.Data().transpose().format(fmt) << ">";
  return os;
}

std::ostream& operator<<(std::ostream& os, const EquirectangularCameraCalf& a) {
  const Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n", "[", "]");
  os << "<EquirectangularCameraCalf " << a.Data().transpose().format(fmt) << ">";
  return os;
}

}  // namespace sym

// Concept implementations for this class
#include "./ops/equirectangular_camera_cal/storage_ops.h"

// Explicit instantiation
template class sym::EquirectangularCameraCal<double>;
template class sym::EquirectangularCameraCal<float>;
