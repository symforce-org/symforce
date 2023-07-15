/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./cc_optimization_stats.h"

#include <pybind11/stl.h>

#include <lcmtypes/sym/optimization_iteration_t.hpp>

#include <symforce/opt/linearization.h>
#include <symforce/opt/optimization_stats.h>
#include <symforce/opt/optional.h>

#include "./lcm_type_casters.h"

namespace sym {

void AddOptimizationStatsWrapper(pybind11::module_ module) {
  py::class_<sym::OptimizationStatsd>(module, "OptimizationStats",
                                      "Debug stats for a full optimization run.")
      .def(py::init<>())
      .def_readwrite("iterations", &sym::OptimizationStatsd::iterations)
      .def_readwrite("best_index", &sym::OptimizationStatsd::best_index,
                     "Index into iterations of the best iteration (containing the optimal Values).")
      .def_readwrite("status", &sym::OptimizationStatsd::status,
                     "What was the result of the optimization? (did it converge, fail, etc.)")
      .def_readwrite("failure_reason", &sym::OptimizationStatsd::failure_reason,
                     "If status == FAILED, why?  This should be cast to the "
                     "NonlinearSolver::FailureReason enum for the nonlinear solver you used.")
      .def_readwrite("jacobian_sparsity", &sym::OptimizationStatsd::jacobian_sparsity,
                     "Sparsity pattern of the problem jacobian (filled out if debug_stats=True)")
      .def_readwrite("linear_solver_ordering", &sym::OptimizationStatsd::linear_solver_ordering,
                     "Ordering used by the linear solver (filled out if debug_stats=True)")
      .def_readwrite("cholesky_factor_sparsity", &sym::OptimizationStatsd::cholesky_factor_sparsity,
                     "Sparsity pattern of the cholesky factor L (filled out if debug_stats=True)")
      .def_property(
          "best_linearization",
          /* getter */
          [](const sym::OptimizationStatsd& stats) -> py::object {
            if (stats.best_linearization) {
              return py::cast(stats.best_linearization.value());
            }
            return py::none();
          },
          /* setter */
          [](sym::OptimizationStatsd& stats,
             const sym::SparseLinearizationd* const best_linearization) {
            if (best_linearization == nullptr) {
              stats.best_linearization = {};
            } else {
              stats.best_linearization = *best_linearization;
            }
          },
          "The linearization at best_index (at optimized_values), filled out if "
          "populate_best_linearization=True")
      .def("get_lcm_type", &sym::OptimizationStatsd::GetLcmType)
      .def(py::pickle(
          [](const sym::OptimizationStatsd& stats) {  //  __getstate__
            return py::make_tuple(
                stats.iterations, stats.best_index, stats.status, stats.failure_reason,
                stats.best_linearization ? py::cast(stats.best_linearization.value()) : py::none());
          },
          [](py::tuple state) {  // __setstate__
            if (state.size() != 5) {
              throw py::value_error("OptimizationStats.__setstate__ expected tuple of size 4.");
            }
            sym::OptimizationStatsd stats;
            stats.iterations = state[0].cast<std::vector<optimization_iteration_t>>();
            stats.best_index = state[1].cast<int32_t>();
            stats.status = state[2].cast<optimization_status_t>();
            stats.failure_reason = state[3].cast<int32_t>();
            const sym::SparseLinearizationd* best_linearization =
                state[4].cast<sym::SparseLinearizationd*>();
            if (best_linearization == nullptr) {
              stats.best_linearization = {};
            } else {
              stats.best_linearization = *best_linearization;
            }
            return stats;
          }));
}

}  // namespace sym
