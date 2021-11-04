#include "./cc_linearization.h"

#include <pybind11/eigen.h>

#include <symforce/opt/linearization.h>

#include "./sym_type_casters.h"

namespace sym {

void AddLinearizationWrapper(pybind11::module_ module) {
  py::class_<sym::Linearizationd>(module, "Linearization")
      .def(py::init<>())
      .def_readwrite("residual", &sym::Linearizationd::residual)
      .def_readwrite("hessian_lower", &sym::Linearizationd::hessian_lower)
      .def_readwrite("jacobian", &sym::Linearizationd::jacobian)
      .def_readwrite("rhs", &sym::Linearizationd::rhs)
      .def("reset", &sym::Linearizationd::Reset)
      .def("is_initialized", &sym::Linearizationd::IsInitialized)
      .def("set_initialized", &sym::Linearizationd::SetInitialized, py::arg("initialized") = true)
      .def("error", &sym::Linearizationd::Error)
      .def("linear_error", &sym::Linearizationd::LinearError, py::arg("x_update"));
}

}  // namespace sym
