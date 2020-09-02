#pragma once

// Import all the known types.
#include <cam/linear_camera_cal.h>
#include <geo/ops/lie_group_ops.h>
#include <geo/ops/storage_ops.h>
#include <geo/pose2.h>
#include <geo/pose3.h>
#include <geo/rot2.h>
#include <geo/rot3.h>

#include <lcmtypes/symforce/type_t.hpp>

namespace sym {

template <typename Scalar>
using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template <typename Scalar>
using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

using type_t = symforce::type_t;

/**
 * Get the enum type value corresponding to the templated type. This function
 * is statically evaluated and becomes the appropriate enum constant.
 */
// TOOD(hayk): We should generate geo::StorageOps<T>::TypeEnum() instead.
template <typename Scalar, typename T>
type_t GetType() {
  if (std::is_same<T, Scalar>::value) {
    return type_t::SCALAR;
  } else if (std::is_same<T, geo::Rot2<Scalar>>::value) {
    return type_t::ROT2;
  } else if (std::is_same<T, geo::Rot3<Scalar>>::value) {
    return type_t::ROT3;
  } else if (std::is_same<T, geo::Pose2<Scalar>>::value) {
    return type_t::POSE2;
  } else if (std::is_same<T, geo::Pose3<Scalar>>::value) {
    return type_t::POSE3;
  } else if (std::is_same<T, Eigen::Matrix<Scalar, 1, 1>>::value) {
    return type_t::VECTOR1;
  } else if (std::is_same<T, Eigen::Matrix<Scalar, 2, 1>>::value) {
    return type_t::VECTOR2;
  } else if (std::is_same<T, Eigen::Matrix<Scalar, 3, 1>>::value) {
    return type_t::VECTOR3;
  } else if (std::is_same<T, Eigen::Matrix<Scalar, 4, 1>>::value) {
    return type_t::VECTOR4;
  } else if (std::is_same<T, Eigen::Matrix<Scalar, 5, 1>>::value) {
    return type_t::VECTOR5;
  } else if (std::is_same<T, Eigen::Matrix<Scalar, 6, 1>>::value) {
    return type_t::VECTOR6;
  } else if (std::is_same<T, Eigen::Matrix<Scalar, 7, 1>>::value) {
    return type_t::VECTOR7;
  } else if (std::is_same<T, Eigen::Matrix<Scalar, 8, 1>>::value) {
    return type_t::VECTOR8;
  } else if (std::is_same<T, Eigen::Matrix<Scalar, 9, 1>>::value) {
    return type_t::VECTOR9;
  } else {
    return type_t::INVALID;
  }
}

/**
 * Helper to handle polymorphism by creating a switch from a runtime type enum to dispatch
 * to the templated method func. Used to perform type-aware operations.
 *
 * Args:
 *   name: Name of the output function (ex: PrintByType)
 *   func: Name of a function template (ex: PrintHelper)
 */
#define BY_TYPE_HELPER(name, func)                  \
  template <typename Scalar, typename... Args>      \
  void name(const type_t type, Args&&... args) {    \
    switch (type.value) {                           \
      case type_t::ROT2:                            \
        func<geo::Rot2<Scalar>>(args...);           \
        break;                                      \
      case type_t::ROT3:                            \
        func<geo::Rot3<Scalar>>(args...);           \
        break;                                      \
      case type_t::POSE2:                           \
        func<geo::Pose2<Scalar>>(args...);          \
        break;                                      \
      case type_t::POSE3:                           \
        func<geo::Pose3<Scalar>>(args...);          \
        break;                                      \
      case type_t::SCALAR:                          \
        func<Scalar>(args...);                      \
        break;                                      \
      case type_t::VECTOR1:                         \
        func<Eigen::Matrix<Scalar, 1, 1>>(args...); \
        break;                                      \
      case type_t::VECTOR2:                         \
        func<Eigen::Matrix<Scalar, 2, 1>>(args...); \
        break;                                      \
      case type_t::VECTOR3:                         \
        func<Eigen::Matrix<Scalar, 3, 1>>(args...); \
        break;                                      \
      case type_t::VECTOR4:                         \
        func<Eigen::Matrix<Scalar, 4, 1>>(args...); \
        break;                                      \
      case type_t::VECTOR5:                         \
        func<Eigen::Matrix<Scalar, 5, 1>>(args...); \
        break;                                      \
      case type_t::VECTOR6:                         \
        func<Eigen::Matrix<Scalar, 6, 1>>(args...); \
        break;                                      \
      case type_t::VECTOR7:                         \
        func<Eigen::Matrix<Scalar, 7, 1>>(args...); \
        break;                                      \
      case type_t::VECTOR8:                         \
        func<Eigen::Matrix<Scalar, 8, 1>>(args...); \
        break;                                      \
      case type_t::VECTOR9:                         \
        func<Eigen::Matrix<Scalar, 9, 1>>(args...); \
        break;                                      \
      default:                                      \
        break;                                      \
    }                                               \
  }

}  // namespace sym
