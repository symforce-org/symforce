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

#include <sym/ops/storage_ops.h>
#include <sym/pose2.h>
#include <sym/pose3.h>
#include <sym/rot2.h>
#include <sym/rot3.h>

namespace py = pybind11;

template <typename T>
constexpr char kPyTypeName[0];
template <>
constexpr char kPyTypeName<sym::Rot2d>[] = "Rot2";
template <>
constexpr char kPyTypeName<sym::Rot3d>[] = "Rot3";
template <>
constexpr char kPyTypeName<sym::Pose2d>[] = "Pose2";
template <>
constexpr char kPyTypeName<sym::Pose3d>[] = "Pose3";

namespace pybind11 {
namespace detail {

// type_caster is what does the conversions between python types and C++ types. Needed
// for custom conversions (which we need for the geo types, since we're not using the wrapper)
template <typename T>
struct sym_type_caster {
  PYBIND11_TYPE_CASTER(T, _(kPyTypeName<T>));

  bool load(const handle src, bool /* implicit_conversion */) {
    // Converts src (a thin wrapper of a PyObject*) to a T, and assigns to value (a member of the
    // class declared by PYBIND11_TYPE_CASTER)
    if (!py::isinstance(src, py::module_::import("sym").attr(kPyTypeName<T>))) {
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
        py::module_::import("sym").attr(kPyTypeName<T>).attr("from_storage");
    py::object result = from_storage(list);
    result.inc_ref();
    return result;
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

}  // namespace detail
}  // namespace pybind11
