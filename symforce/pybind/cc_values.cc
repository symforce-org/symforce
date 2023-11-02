/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./cc_values.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
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
#include <sym/unit3.h>
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

template <typename Scalar>
py::object PyAtMatrix(const sym::Valuesd& v, const sym::index_entry_t& index_entry) {
  const auto shape = EigenTypeShape(index_entry.type);
  if (shape.first == 1 || shape.second == 1) {
    return py::cast(Eigen::Map<const Eigen::VectorXd>(v.Data().data() + index_entry.offset,
                                                      shape.first * shape.second),
                    py::return_value_policy::copy);
  } else {
    return py::cast(Eigen::Map<const Eigen::MatrixXd>(v.Data().data() + index_entry.offset,
                                                      shape.first, shape.second),
                    py::return_value_policy::copy);
  }
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
BY_TYPE_HELPER(DynamicPyAt, PyAt, PyAtMatrix)

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
          py::arg("key"), py::arg("value"),
          "Add or update a value by key. Returns true if added, false if updated.");
  cls.def("set", py::overload_cast<const sym::index_entry_t&, const T&>(&sym::Valuesd::Set<T>),
          py::arg("key"), py::arg("value"),
          "Update a value by index entry with no map lookup (compared to Set(key)). This does NOT "
          "add new values and assumes the key exists already.");
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
  static void Register(py::class_<sym::Valuesd> /* cls */) {}
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
  auto values_class = py::class_<sym::Valuesd>(module, "Values", R"(
    Efficient polymorphic data structure to store named types with a dict-like interface and
    support efficient repeated operations using a key index. Supports on-manifold optimization.

    Compatible types are given by the type_t enum. All types implement the StorageOps and
    LieGroupOps concepts, which are the core operating mechanisms in this class.
  )");
  values_class.def(py::init<>(), "Default construct as empty.")
      .def(py::init<const sym::values_t&>(), py::arg("msg"), "Construct from serialized form.")
      .def("has", &sym::Valuesd::Has, py::arg("key"), "Return whether the key exists.")
      .def("at", &ValuesAt, py::arg("key"), "Retrieve a value by key.")
      .def("update_or_set", &sym::Valuesd::UpdateOrSet, py::arg("index"), py::arg("other"), R"(
          Update or add keys to this Values base on other Values of different structure.
          index MUST be valid for other.

          NOTE(alvin): it is less efficient than the Update methods below if index objects are created and cached. This method performs map lookup for each key of the index
      )")
      .def("num_entries", &sym::Valuesd::NumEntries, "Number of keys.")
      .def("empty", &sym::Valuesd::Empty, "Has zero keys.")
      .def("keys", &sym::Valuesd::Keys, py::arg("sort_by_offset") = true, R"(
          Get all keys.

          Args:
              sort_by_offset: Sorts by storage order to make iteration safer and more memory efficient
      )")
      .def("items", &sym::Valuesd::Items, "Expose map type to allow iteration.")
      .def("data", py::overload_cast<>(&sym::Valuesd::Data, py::const_), "Raw data buffer.")
      .def("remove", &sym::Valuesd::Remove, py::arg("key"), R"(
          Remove the given key. Only removes the index entry, does not change the data array.
          Returns true if removed, false if already not present.

          Call cleanup() to re-pack the data array.
      )")
      .def("remove_all", &sym::Valuesd::RemoveAll, "Remove all keys and empty out the storage.")
      .def("cleanup", &sym::Valuesd::Cleanup, R"(
          Repack the data array to get rid of empty space from removed keys. If regularly removing
          keys, it's up to the user to call this appropriately to avoid storage growth. Returns the
          number of Scalar elements cleaned up from the data array.

          It will INVALIDATE all indices, offset increments, and pointers.
          Re-create an index with create_index().
      )")
      .def("create_index", &sym::Valuesd::CreateIndex, py::arg("keys"), R"(
          Create an index from the given ordered subset of keys. This object can then be used
          for repeated efficient operations on that subset of keys.

          If you want an index of all the keys, call `values.create_index(values.keys())`.

          An index will be INVALIDATED if the following happens:
            1) remove() is called with a contained key, or remove_all() is called
            2) cleanup() is called to re-pack the data array
      )")
      .def("at", &ValuesAtIndexEntry, py::arg("entry"),
           "Retrieve a value by index entry. This avoids a map lookup compared to at(key).")
      .def("update",
           py::overload_cast<const sym::index_t&, const sym::Valuesd&>(&sym::Valuesd::Update),
           py::arg("index"), py::arg("other"),
           "Efficiently update the keys given by this index from other into this. This purely "
           "copies slices of the data arrays, the index MUST be valid for both objects!")
      .def("update",
           py::overload_cast<const sym::index_t&, const sym::index_t&, const sym::Valuesd&>(
               &sym::Valuesd::Update),
           py::arg("index_this"), py::arg("index_other"), py::arg("other"),
           "Efficiently update the keys from a different structured Values, given by this index "
           "and other index. This purely copies slices of the data arrays. index_this MUST be "
           "valid for this object; index_other MUST be valid for other object.")
      .def(
          "retract",
          [](sym::Valuesd& v, const sym::index_t& index, const std::vector<double>& delta,
             const double epsilon) {
            if (index.tangent_dim != static_cast<int>(delta.size())) {
              throw std::runtime_error(
                  fmt::format("The length of delta [{}] must match index.tangent_dim [{}]",
                              delta.size(), index.tangent_dim));
            }
            v.Retract(index, delta.data(), epsilon);
          },
          py::arg("index"), py::arg("delta"), py::arg("epsilon"), R"(
              Perform a retraction from an update vector.

              Args:
                  index: Ordered list of keys in the delta vector
                  delta: Update vector - MUST be the size of index.tangent_dim!
                  epsilon: Small constant to avoid singularities (do not use zero)
          )")
      .def("local_coordinates", &sym::Valuesd::LocalCoordinates, py::arg("others"),
           py::arg("index"), py::arg("epsilon"), R"(
          Express this Values in the local coordinate of others Values, i.e., this \ominus others

          Args:
              others: The other Values that the local coordinate is relative to
              index: Ordered list of keys to include (MUST be valid for both this and others Values)
              epsilon: Small constant to avoid singularities (do not use zero)
           )")
      .def("get_lcm_type", &sym::Valuesd::GetLcmType, py::arg("sort_keys") = false,
           "Serialize to LCM.")
      .def("__repr__", [](const sym::Valuesd& values) { return fmt::format("{}", values); })
      .def(py::pickle(
          [](const sym::Valuesd& values) {  //  __getstate__
            const sym::values_t lcm_values = values.GetLcmType();
            const auto encoded_size = lcm_values.getEncodedSize();
            std::vector<char> buffer(encoded_size);
            const auto encoded_bytes = lcm_values.encode(buffer.data(), 0, encoded_size);
            if (encoded_bytes < 0) {
              throw std::runtime_error("An error occured while encoded a Values object.");
            }
            return py::bytes(buffer.data(), encoded_bytes);
          },
          [](py::bytes state) {  // __setstate__
            const std::string buffer = state.cast<std::string>();
            sym::values_t lcm_values;
            const auto decoded_bytes = lcm_values.decode(buffer.data(), 0, buffer.size());
            if (decoded_bytes < 0) {
              throw std::runtime_error("An error occured while decoding a Values object.");
            }
            return sym::Valuesd(lcm_values);
          }));
  RegisterTypeWithValues<double>(values_class);
  RegisterTypeWithValues<sym::Rot2d>(values_class);
  RegisterTypeWithValues<sym::Rot3d>(values_class);
  RegisterTypeWithValues<sym::Pose2d>(values_class);
  RegisterTypeWithValues<sym::Pose3d>(values_class);
  RegisterTypeWithValues<sym::Unit3d>(values_class);
  RegisterTypeWithValues<sym::ATANCameraCald>(values_class);
  RegisterTypeWithValues<sym::DoubleSphereCameraCald>(values_class);
  RegisterTypeWithValues<sym::EquirectangularCameraCald>(values_class);
  RegisterTypeWithValues<sym::LinearCameraCald>(values_class);
  RegisterTypeWithValues<sym::PolynomialCameraCald>(values_class);
  RegisterTypeWithValues<sym::SphericalCameraCald>(values_class);
  // The template paramater below is 9 because all (and only) matrices up to size 9x9 are supported
  // by sym::Values.
  RegisterMatrices<9>(values_class);
}

}  // namespace sym
