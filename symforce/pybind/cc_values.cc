#include "./cc_values.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <lcmtypes/sym/index_entry_t.hpp>
#include <lcmtypes/sym/index_t.hpp>
#include <lcmtypes/sym/type_t.hpp>
#include <lcmtypes/sym/values_t.hpp>

#include <sym/ops/storage_ops.h>
#include <sym/pose2.h>
#include <sym/pose3.h>
#include <sym/rot2.h>
#include <sym/rot3.h>
#include <sym/util/type_ops.h>
#include <symforce/opt/key.h>
#include <symforce/opt/values.h>

#include "./lcm_type_casters.h"
#include "./sym_type_casters.h"

namespace py = pybind11;

namespace sym {

//================================================================================================//
//---------------------------- Helpers for wrapping sym::Valuesd ---------------------------------//
//================================================================================================//

/**
 * Calls v.At<T>(index_entry) and casts the return value to a py::object.
 *
 * This function exists so that all template specializations of Valuesd::At<T>
 * can be referenced with a common signature (to ease python wrapping).
 */
template <typename T>
py::object PyAt(const sym::Valuesd& v, const sym::index_entry_t& index_entry) {
  return py::cast(v.At<T>(index_entry));
}

/**
 * Has signature
 * template <typename Scalar>
 * py::object DynamicPyAt(const type_t type, const Valuesd& v, const index_entry_t& index_entry);
 *
 * For supported types (see macro definition), identifies the type T represented by type, then
 * returns PyAt<T>(v, index_entry).
 *
 * Precondition: type is a supported type_t
 */
BY_TYPE_HELPER(DynamicPyAt, PyAt)

/**
 * Dynamically identifies the type T stored in v at index_entry, then returns
 * v.At<T>(index_entry) casted to a py::object.
 */
py::object ValuesAtIndexEntry(const sym::Valuesd& v, const sym::index_entry_t& index_entry) {
  return DynamicPyAt<double>(index_entry.type, v, index_entry);
}

/**
 * Identifies the index_entry_t entry in v indexed by key, dynamically identifies the type T stored
 * in v at index_entry, then returns
 * v.At<T>(index_entry) casted to a py::object.
 *
 * Precondition:
 * A value in v has Key key
 */
py::object ValuesAt(const sym::Valuesd& v, const sym::Key& key) {
  sym::index_entry_t index_entry = v.IndexEntryAt(key);
  return ValuesAtIndexEntry(v, index_entry);
}

/**
 * Registers the set methods of Valuesd with a python wrapper of the class for the template
 * specializations of T.
 *
 * This function enables v.set(ob) to work for v an instance of a wrapped Valuesd class, and ob
 * an instance of a class which can be casted to a T by pybind11.
 */
template <typename T>
void RegisterTypeWithValues(py::class_<sym::Valuesd> cls) {
  cls.def("set", py::overload_cast<const sym::Key&, const T&>(&sym::Valuesd::Set<T>),
          py::arg("key"), py::arg("value"));
  cls.def("set", py::overload_cast<const sym::index_entry_t&, const T&>(&sym::Valuesd::Set<T>),
          py::arg("key"), py::arg("value"));
}

/**
 * Calls RegisterTypeWithValues<Eigen::Matrix<double, n, m>>(cls) for
 * all n in [1, N] and m in [1, M]
 */
template <int N, int M>
struct RegisterMatricesHelper {
  static void Register(py::class_<sym::Valuesd> cls) {
    RegisterTypeWithValues<Eigen::Matrix<double, N, M>>(cls);
    if (N != M) {
      RegisterTypeWithValues<Eigen::Matrix<double, M, N>>(cls);
    }
    RegisterMatricesHelper<N - 1, M>::Register(cls);
  }
};

template <int M>
struct RegisterMatricesHelper<0, M> {
  static void Register(py::class_<sym::Valuesd> cls) {
    RegisterMatricesHelper<M - 1, M - 1>::Register(cls);
  }
};

template <>
struct RegisterMatricesHelper<0, 1> {
  static void Register(py::class_<sym::Valuesd> cls) {}
};

/**
 * Calls RegisterTypeWithValues<Eigen::Matrix<double< n, m>>(cls) for all
 * n, m in [1, SquareSize]
 */
template <int SquareSize>
constexpr void RegisterMatrices(py::class_<sym::Valuesd>& cls) {
  RegisterMatricesHelper<SquareSize, SquareSize>::Register(cls);
}

//================================================================================================//
//-------------------------------- The Public Values Wrapper -------------------------------------//
//================================================================================================//

void AddValuesWrapper(pybind11::module_ module) {
  auto values_class = py::class_<sym::Valuesd>(module, "Values");
  values_class.def(py::init<>())
      .def(py::init<const sym::values_t&>(), py::arg("msg"))
      .def("has", &sym::Valuesd::Has, py::arg("key"))
      .def("at", &ValuesAt, py::arg("key"))
      .def("update_or_set", &sym::Valuesd::UpdateOrSet, py::arg("index"), py::arg("other"))
      .def("num_entries", &sym::Valuesd::NumEntries)
      .def("empty", &sym::Valuesd::Empty)
      .def("keys", &sym::Valuesd::Keys, py::arg("sort_by_offset") = true)
      .def("items", &sym::Valuesd::Items)
      .def("data", &sym::Valuesd::Data)
      .def("remove", &sym::Valuesd::Remove, py::arg("key"))
      .def("remove_all", &sym::Valuesd::RemoveAll)
      .def("cleanup", &sym::Valuesd::Cleanup)
      .def("create_index", &sym::Valuesd::CreateIndex, py::arg("keys"))
      .def("at", &ValuesAtIndexEntry, py::arg("entry"))
      .def("update",
           py::overload_cast<const sym::index_t&, const sym::Valuesd&>(&sym::Valuesd::Update),
           py::arg("index"), py::arg("other"))
      .def("update",
           py::overload_cast<const sym::index_t&, const sym::index_t&, const sym::Valuesd&>(
               &sym::Valuesd::Update),
           py::arg("index_this"), py::arg("index_other"), py::arg("other"))
      .def(
          "retract",
          [](sym::Valuesd& v, const sym::index_t& index, const std::vector<double>& delta,
             const double epsilon) {
            if (index.tangent_dim != delta.size()) {
              throw std::runtime_error(
                  fmt::format("The length of delta [{}] must match index.tangent_dim [{}]",
                              delta.size(), index.tangent_dim));
            }
            v.Retract(index, delta.data(), epsilon);
          },
          py::arg("index"), py::arg("delta"), py::arg("epsilon"))
      .def("local_coordinates", &sym::Valuesd::LocalCoordinates, py::arg("others"),
           py::arg("index"), py::arg("epsilon"))
      .def("get_lcm_type", &sym::Valuesd::GetLcmType)
      .def("__repr__", [](const sym::Valuesd& values) { return fmt::format("{}", values); });
  RegisterTypeWithValues<double>(values_class);
  RegisterTypeWithValues<sym::Rot2d>(values_class);
  RegisterTypeWithValues<sym::Rot3d>(values_class);
  RegisterTypeWithValues<sym::Pose2d>(values_class);
  RegisterTypeWithValues<sym::Pose3d>(values_class);
  // The template paramater below is 9 because all (and only) matrices up to size 9x9 are supported
  // by sym::Values.
  RegisterMatrices<9>(values_class);
}

}  // namespace sym
