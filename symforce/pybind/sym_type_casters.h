/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

// -----------------------------------------------------------------------------
// Defines template specializations of pybind11::detail::type_caster<T> for
// T = sym::Rot2, sym::Rot3, sym::Pose2, and sym::Pose3.
//
// Must be included in any file using pybind11 to wrap functions whose argument
// types or return types are any of the above geometric types.
// -----------------------------------------------------------------------------

#pragma once

#include <fmt/format.h>
#include <pybind11/pybind11.h>

#include <sym/atan_camera_cal.h>
#include <sym/double_sphere_camera_cal.h>
#include <sym/equirectangular_camera_cal.h>
#include <sym/linear_camera_cal.h>
#include <sym/ops/storage_ops.h>
#include <sym/polynomial_camera_cal.h>
#include <sym/pose2.h>
#include <sym/pose3.h>
#include <sym/rot2.h>
#include <sym/rot3.h>
#include <sym/spherical_camera_cal.h>

namespace py = pybind11;

namespace pybind11 {
namespace detail {

// based on pybind11/cast.h:783
template <typename T>
struct handle_sym_type_name {
  static constexpr auto name = _("");
};

template <>
struct handle_sym_type_name<sym::Rot2d> {
  static constexpr auto name = _("Rot2");
};

template <>
struct handle_sym_type_name<sym::Rot3d> {
  static constexpr auto name = _("Rot3");
};

template <>
struct handle_sym_type_name<sym::Pose2d> {
  static constexpr auto name = _("Pose2");
};

template <>
struct handle_sym_type_name<sym::Pose3d> {
  static constexpr auto name = _("Pose3");
};

template <>
struct handle_sym_type_name<sym::ATANCameraCald> {
  static constexpr auto name = _("ATANCameraCal");
};
template <>
struct handle_sym_type_name<sym::DoubleSphereCameraCald> {
  static constexpr auto name = _("DoubleSphereCameraCal");
};

template <>
struct handle_sym_type_name<sym::EquirectangularCameraCald> {
  static constexpr auto name = _("EquirectangularCameraCal");
};

template <>
struct handle_sym_type_name<sym::LinearCameraCald> {
  static constexpr auto name = _("LinearCameraCal");
};

template <>
struct handle_sym_type_name<sym::PolynomialCameraCald> {
  static constexpr auto name = _("PolynomialCameraCal");
};

template <>
struct handle_sym_type_name<sym::SphericalCameraCald> {
  static constexpr auto name = _("SphericalCameraCal");
};

// type_caster is what does the conversions between python types and C++ types. Needed
// for custom conversions (which we need for the geo types, since we're not using the wrapper)
template <typename T>
struct sym_type_caster {
  PYBIND11_TYPE_CASTER(T, handle_sym_type_name<T>::name);

  sym_type_caster() : value(T::DataVec::Zero()) {}

  bool load(const handle src, bool /* implicit_conversion */) {
    // Converts src (a thin wrapper of a PyObject*) to a T, and assigns to value (a member of the
    // class declared by PYBIND11_TYPE_CASTER)
    if (!py::isinstance(src, py::module_::import("sym").attr(handle_sym_type_name<T>::name.text))) {
      return false;
    }
    const std::vector<double> data_vec = src.attr("to_storage")().cast<std::vector<double>>();
    value = T(Eigen::Matrix<double, sym::StorageOps<T>::StorageDim(), 1>(data_vec.data()));
    return true;
  }

  static handle cast(const T src, return_value_policy /* policy */, handle /* parent */) {
    // Constructs and returns a py::object representing the same data as src
    constexpr auto storage_dim = sym::StorageOps<T>::StorageDim();
    py::list list(storage_dim);
    const auto data = src.Data();
    for (int i = 0; i < storage_dim; i++) {
      list[i] = data[i];
    }
    const py::object from_storage =
        py::module_::import("sym").attr(handle_sym_type_name<T>::name.text).attr("from_storage");
    py::object result = from_storage(list);
    result.inc_ref();
    return std::move(result);
  }
};

// Defining type_caster<T> for the sym types
template <>
struct type_caster<sym::Rot2d> : public sym_type_caster<sym::Rot2d> {};
template <>
struct type_caster<sym::Rot3d> : public sym_type_caster<sym::Rot3d> {};
template <>
struct type_caster<sym::Pose2d> : public sym_type_caster<sym::Pose2d> {};
template <>
struct type_caster<sym::Pose3d> : public sym_type_caster<sym::Pose3d> {};
template <>
struct type_caster<sym::ATANCameraCald> : public sym_type_caster<sym::ATANCameraCald> {};
template <>
struct type_caster<sym::DoubleSphereCameraCald>
    : public sym_type_caster<sym::DoubleSphereCameraCald> {};
template <>
struct type_caster<sym::EquirectangularCameraCald>
    : public sym_type_caster<sym::EquirectangularCameraCald> {};
template <>
struct type_caster<sym::LinearCameraCald> : public sym_type_caster<sym::LinearCameraCald> {};
template <>
struct type_caster<sym::PolynomialCameraCald> : public sym_type_caster<sym::PolynomialCameraCald> {
};
template <>
struct type_caster<sym::SphericalCameraCald> : public sym_type_caster<sym::SphericalCameraCald> {};

}  // namespace detail
}  // namespace pybind11
