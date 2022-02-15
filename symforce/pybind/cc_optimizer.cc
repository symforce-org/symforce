/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./cc_optimizer.h"

#include <vector>

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <lcmtypes/sym/optimizer_params_t.hpp>

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
  py::class_<Optimizerd>(module, "Optimizer")
      .def(py::init<const optimizer_params_t&, const std::vector<Factord>&, const double,
                    const std::string&, const std::vector<Key>&, bool, bool>(),
           py::arg("params"), py::arg("factors"), py::arg("epsilon") = 1e-9,
           py::arg("name") = "sym::Optimize", py::arg("keys") = std::vector<Key>(),
           py::arg("debug_stats") = false, py::arg("check_derivatives") = false)
      .def("optimize", py::overload_cast<Valuesd*, int, bool>(&Optimizerd::Optimize),
           py::arg("values"), py::arg("num_iterations") = -1,
           py::arg("populate_best_linearization") = false)
      .def("optimize",
           py::overload_cast<Valuesd*, int, bool, OptimizationStatsd*>(&Optimizerd::Optimize),
           py::arg("values"), py::arg("num_iterations"), py::arg("populate_best_linearization"),
           py::arg("stats"))
      .def("optimize", py::overload_cast<Valuesd*, int, OptimizationStatsd*>(&Optimizerd::Optimize),
           py::arg("values"), py::arg("num_iterations"), py::arg("stats"))
      .def("optimize", py::overload_cast<Valuesd*, OptimizationStatsd*>(&Optimizerd::Optimize),
           py::arg("values"), py::arg("stats"))
      .def("linearize", &Optimizerd::Linearize, py::arg("values"))
      .def(
          "compute_all_covariances",
          [](Optimizerd& opt, const Linearizationd& linearization) {
            std::unordered_map<Key, Eigen::MatrixXd> covariances_by_key;
            opt.ComputeAllCovariances(linearization, &covariances_by_key);
            return covariances_by_key;
          },
          py::arg("linearization"))
      .def(
          "compute_covariances",
          [](Optimizerd& opt, const Linearizationd& linearization, const std::vector<Key>& keys) {
            std::unordered_map<Key, Eigen::MatrixXd> covariances_by_key;
            opt.ComputeCovariances(linearization, keys, &covariances_by_key);
            return covariances_by_key;
          },
          py::arg("linearization"), py::arg("keys"))
      .def("keys", &Optimizerd::Keys)
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
          py::arg("key"))
      .def("update_params", &Optimizerd::UpdateParams, py::arg("params"));

  // Wrapping free functions
  module.def("optimize", &Optimize<double>, py::arg("params"), py::arg("factors"),
             py::arg("values"), py::arg("epsilon") = 1e-9);
  module.def("default_optimizer_params", &DefaultOptimizerParams);
}

}  // namespace sym
