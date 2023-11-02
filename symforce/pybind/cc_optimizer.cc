/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./cc_optimizer.h"

#include <vector>

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <lcmtypes/sym/optimizer_params_t.hpp>

#include <sym/util/epsilon.h>
#include <symforce/opt/factor.h>
#include <symforce/opt/key.h>
#include <symforce/opt/linearization.h>
#include <symforce/opt/optimization_stats.h>
#include <symforce/opt/optimizer.h>
#include <symforce/opt/values.h>

#include "./lcm_type_casters.h"
#include "./sym_type_casters.h"

namespace sym {

void AddOptimizerWrapper(pybind11::module_ module) {
  py::class_<Optimizerd>(module, "Optimizer",
                         "Class for optimizing a nonlinear least-squares problem specified as a "
                         "list of Factors. For efficient use, create once and call Optimize() "
                         "multiple times with different initial guesses, as long as the factors "
                         "remain constant and the structure of the Values is identical.")
      .def(py::init<const optimizer_params_t&, const std::vector<Factord>&, const double,
                    const std::string&, const std::vector<Key>&, bool, bool, bool>(),
           py::arg("params"), py::arg("factors"), py::arg("epsilon") = kDefaultEpsilond,
           py::arg("name") = "sym::Optimize", py::arg("keys") = std::vector<Key>(),
           py::arg("debug_stats") = false, py::arg("check_derivatives") = false,
           py::arg("include_jacobians") = false)
      .def("optimize", py::overload_cast<Valuesd&, int, bool>(&Optimizerd::Optimize),
           py::arg("values"), py::arg("num_iterations") = -1,
           py::arg("populate_best_linearization") = false, R"(
           Optimize the given values in-place

           Args:
             num_iterations: If < 0 (the default), uses the number of iterations specified by the params at construction.

             populate_best_linearization: If true, the linearization at the best values will be filled out in the stats.

           Returns:
               The optimization stats
           )")
      .def("optimize",
           py::overload_cast<Valuesd&, int, bool, OptimizationStatsd&>(&Optimizerd::Optimize),
           py::arg("values"), py::arg("num_iterations"), py::arg("populate_best_linearization"),
           py::arg("stats"), R"(
           Optimize the given values in-place

           This overload takes the stats as an argument, and stores into there.  This allows users to
           avoid reallocating memory for any of the entries in the stats, for use cases where that's
           important.  If passed, stats must not be None.

           Args:
             num_iterations: If < 0 (the default), uses the number of iterations specified by the params at construction

             populate_best_linearization: If true, the linearization at the best values will be filled out in the stats

             stats: An OptimizationStats to fill out with the result - if filling out dynamically allocated fields here, will not reallocate if memory is already allocated in the required shape (e.g. for repeated calls to Optimize)
           )")
      .def("optimize", py::overload_cast<Valuesd&, int, OptimizationStatsd&>(&Optimizerd::Optimize),
           py::arg("values"), py::arg("num_iterations"), py::arg("stats"), R"(
           Optimize the given values in-place

           This overload takes the stats as an argument, and stores into there.  This allows users to
           avoid reallocating memory for any of the entries in the stats, for use cases where that's
           important.  If passed, stats must not be None.

           Args:
             num_iterations: If < 0 (the default), uses the number of iterations specified by the params at construction

             stats: An OptimizationStats to fill out with the result - if filling out dynamically allocated fields here, will not reallocate if memory is already allocated in the required shape (e.g. for repeated calls to Optimize)
           )")
      .def("optimize", py::overload_cast<Valuesd&, OptimizationStatsd&>(&Optimizerd::Optimize),
           py::arg("values"), py::arg("stats"), R"(
           Optimize the given values in-place

           This overload takes the stats as an argument, and stores into there.  This allows users to
           avoid reallocating memory for any of the entries in the stats, for use cases where that's
           important.  If passed, stats must not be None.

           Args:
             stats: An OptimizationStats to fill out with the result - if filling out dynamically allocated fields here, will not reallocate if memory is already allocated in the required shape (e.g. for repeated calls to Optimize)
           )")
      .def("linearize", &Optimizerd::Linearize, py::arg("values"),
           "Linearize the problem around the given values.")
      .def(
          "compute_all_covariances",
          [](Optimizerd& opt, const SparseLinearizationd& linearization) {
            std::unordered_map<Key, Eigen::MatrixXd> covariances_by_key;
            opt.ComputeAllCovariances(linearization, covariances_by_key);
            return covariances_by_key;
          },
          py::arg("linearization"), R"(
          Get covariances for each optimized key at the given linearization

          May not be called before either optimize or linearize has been called.
          )")
      .def(
          "compute_covariances",
          [](Optimizerd& opt, const SparseLinearizationd& linearization,
             const std::vector<Key>& keys) {
            std::unordered_map<Key, Eigen::MatrixXd> covariances_by_key;
            opt.ComputeCovariances(linearization, keys, covariances_by_key);
            return covariances_by_key;
          },
          py::arg("linearization"), py::arg("keys"), R"(
          Get covariances for the given subset of keys at the given linearization

          This version is potentially much more efficient than computing the covariances for all
          keys in the problem.

          Currently requires that `keys` corresponds to a set of keys at the start of the list of keys
          for the full problem, and in the same order.  It uses the Schur complement trick, so will be
          most efficient if the hessian is of the following form, with C block diagonal::

              A = ( B    E )
                  ( E^T  C )
          )")
      .def(
          "compute_full_covariance",
          [](Optimizerd& opt, const SparseLinearizationd& linearization) {
            Eigen::MatrixXd covariance;
            opt.ComputeFullCovariance(linearization, covariance);
            return covariance;
          },
          py::arg("linearization"), R"(
          Get the full problem covariance at the given linearization

          Unlike compute_covariance and compute_all_covariances, this includes the off-diagonal
          blocks, i.e. the cross-covariances between different keys.

          The ordering of entries here is the same as the ordering of the keys in the linearization,
          which can be accessed via linearization_index().

          May not be called before either optimize or linearize has been called.
          )")
      .def("keys", &Optimizerd::Keys, "Get the optimized keys.")
      .def("factors", &Optimizerd::Factors, "Get the factors.")
      .def("update_params", &Optimizerd::UpdateParams, py::arg("params"),
           "Update the optimizer params.")
      .def("linearization_index",
           [](const Optimizerd& opt) -> py::dict {
             // Convert to cc_sym.Key, which is hashable
             py::dict py_index;
             for (const auto& key_and_entry : opt.Linearizer().StateIndex()) {
               py_index[py::cast(Key(key_and_entry.first))] = key_and_entry.second;
             }
             return py_index;
           })
      .def(
          "linearization_index_entry",
          [](const Optimizerd& opt, const Key& key) -> index_entry_t {
            return opt.Linearizer().StateIndex().at(key.GetLcmType());
          },
          py::arg("key"));

  // Wrapping free functions
  // NOTE(brad): the overload cast is only necessary because we temporarily have two overloads,
  // one for the new signature, and one for a deprecated signature. The overload cast could be
  // removed once the deprecated overload is removed.
  module.def("optimize",
             py::overload_cast<const optimizer_params_t&, const std::vector<Factord>&, Valuesd&,
                               const double>(&Optimize<double>),
             py::arg("params"), py::arg("factors"), py::arg("values"),
             py::arg("epsilon") = kDefaultEpsilond,
             "Simple wrapper to make optimization one function call.");
  module.def("default_optimizer_params", &DefaultOptimizerParams,
             "Sensible default parameters for Optimizer.");
}

}  // namespace sym
