/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./cc_linearization.h"

#include <pybind11/eigen.h>

#include <symforce/opt/linearization.h>

#include "./sym_type_casters.h"

namespace py = pybind11;

namespace sym {

void AddLinearizationWrapper(pybind11::module_ module) {
  py::class_<sym::Linearizationd>(module, "Linearization",
                                  "Class for storing a problem linearization evaluated at a Values "
                                  "(i.e. a residual, jacobian, hessian, and rhs).")
      .def(py::init<>())
      .def_readwrite("residual", &sym::Linearizationd::residual)
      .def_readwrite("hessian_lower", &sym::Linearizationd::hessian_lower)
      .def_readwrite("jacobian", &sym::Linearizationd::jacobian)
      .def_readwrite("rhs", &sym::Linearizationd::rhs)
      .def("reset", &sym::Linearizationd::Reset, "Set to invalid.")
      .def("is_initialized", &sym::Linearizationd::IsInitialized,
           "Returns whether the linearization is currently valid for the corresponding values. "
           "Accessing any of the members when this is false could result in unexpected behavior.")
      .def("set_initialized", &sym::Linearizationd::SetInitialized, py::arg("initialized") = true)
      .def("error", &sym::Linearizationd::Error)
      .def("linear_error", &sym::Linearizationd::LinearError, py::arg("x_update"));
}

}  // namespace sym
