#include "./cc_optimization_stats.h"

#include <pybind11/stl.h>

#include <symforce/opt/linearization.h>
#include <symforce/opt/optimization_stats.h>

#include "./lcm_type_casters.h"

namespace sym {

void AddOptimizationStatsWrapper(pybind11::module_ module) {
  py::class_<sym::OptimizationStatsd>(module, "OptimizationStats")
      .def(py::init<>())
      .def_readwrite("iterations", &sym::OptimizationStatsd::iterations)
      .def_readwrite("best_index", &sym::OptimizationStatsd::best_index)
      .def_readwrite("early_exited", &sym::OptimizationStatsd::early_exited)
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
          [](sym::OptimizationStatsd& stats, const sym::Linearizationd* const best_linearization) {
            if (best_linearization == nullptr) {
              stats.best_linearization = {};
            } else {
              stats.best_linearization = *best_linearization;
            }
          })
      .def("get_lcm_type", &sym::OptimizationStatsd::GetLcmType);
};

}  // namespace sym
