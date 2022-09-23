/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

// -----------------------------------------------------------------------------
// Defines template specializations of pybind11::detail::type_caster<T> for
// various lcm types.
//
// Must be included in any file using pybind11 to wrap functions whose argument
// types or return types are any of specialized lcm types.
// -----------------------------------------------------------------------------

#pragma once

#include <fmt/format.h>
#include <pybind11/pybind11.h>

#include <lcmtypes/sym/index_entry_t.hpp>
#include <lcmtypes/sym/index_t.hpp>
#include <lcmtypes/sym/key_t.hpp>
#include <lcmtypes/sym/linearized_dense_factor_t.hpp>
#include <lcmtypes/sym/optimization_iteration_t.hpp>
#include <lcmtypes/sym/optimization_stats_t.hpp>
#include <lcmtypes/sym/optimizer_params_t.hpp>
#include <lcmtypes/sym/type_t.hpp>
#include <lcmtypes/sym/values_t.hpp>

namespace py = pybind11;

namespace pybind11 {
namespace detail {

template <typename LCMType>
struct lcm_type_caster {
  PYBIND11_TYPE_CASTER(LCMType, _(*LCMType::getTypeNameArrayPtr()));

  bool load(const handle src, bool /* implicit_conversion */) {
    // Converts src (a thin wrapper of a PyObject*) to a T, and assigns to value (a member of the
    // class declared by PYBIND11_TYPE_CASTER)

    // An alternative (which I think I'd like to go with), is to explictly check if
    // the src is an instance of the actual type.
    const std::string name = src.attr("__class__").attr("__name__").cast<std::string>();
    if (name != LCMType::getTypeName()) {
      return false;
    }
    const py::object encoded_msg_bytes = src.attr("encode")();
    const char* encoded_msg = PyBytes_AsString(encoded_msg_bytes.ptr());
    const size_t msg_len = py::len(encoded_msg_bytes);
    const auto ret = value.decode(encoded_msg, 0, msg_len);
    if (ret < 0) {
      throw std::runtime_error(
          fmt::format("Failed to decode {} (data length: {}", LCMType::getTypeName(), msg_len));
    }
    return true;
  }

  static handle cast(const LCMType src, return_value_policy /* policy */, handle /* parent */) {
    // Constructs and returns a py::object representing the same data as src
    const auto msg_len = src.getEncodedSize();
    std::vector<char> msg_buf(msg_len);
    const auto ret = src.encode(msg_buf.data(), 0, msg_len);
    if (ret < 0) {
      throw std::runtime_error(
          fmt::format("Failed to encode {} (data length: {}", LCMType::getTypeName(), msg_len));
    }
    const py::bytes msg_bytes(msg_buf.data(), msg_len);
    const std::string module_path =
        fmt::format("lcmtypes.{}._{}", LCMType::getPackageName(), LCMType::getTypeName());
    const py::object lcm_py_type =
        py::module_::import(module_path.c_str()).attr(LCMType::getTypeName());
    py::object result = lcm_py_type.attr("decode")(msg_bytes);
    result.inc_ref();
    return std::move(result);
  }
};

// Defining type_caster<T> for the lcm types
template <>
struct type_caster<sym::index_entry_t> : public lcm_type_caster<sym::index_entry_t> {};
template <>
struct type_caster<sym::index_t> : public lcm_type_caster<sym::index_t> {};
template <>
struct type_caster<sym::key_t> : public lcm_type_caster<sym::key_t> {};
template <>
struct type_caster<sym::linearized_dense_factor_t>
    : public lcm_type_caster<sym::linearized_dense_factor_t> {};
template <>
struct type_caster<sym::optimization_iteration_t>
    : public lcm_type_caster<sym::optimization_iteration_t> {};
template <>
struct type_caster<sym::optimization_stats_t> : public lcm_type_caster<sym::optimization_stats_t> {
};
template <>
struct type_caster<sym::optimizer_params_t> : public lcm_type_caster<sym::optimizer_params_t> {};
template <>
struct type_caster<sym::values_t> : public lcm_type_caster<sym::values_t> {};

}  // namespace detail
}  // namespace pybind11
