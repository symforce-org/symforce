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
  py::class_<sym::SparseLinearizationd>(
      module, "Linearization",
      "Class for storing a problem linearization evaluated at a Values "
      "(i.e. a residual, jacobian, hessian, and rhs).")
      .def(py::init<>())
      .def_readwrite("residual", &sym::SparseLinearizationd::residual)
      .def_readwrite("hessian_lower", &sym::SparseLinearizationd::hessian_lower)
      .def_readwrite("jacobian", &sym::SparseLinearizationd::jacobian)
      .def_readwrite("rhs", &sym::SparseLinearizationd::rhs)
      .def("reset", &sym::SparseLinearizationd::Reset, "Set to invalid.")
      .def("is_initialized", &sym::SparseLinearizationd::IsInitialized,
           "Returns whether the linearization is currently valid for the corresponding values. "
           "Accessing any of the members when this is false could result in unexpected behavior.")
      .def("set_initialized", &sym::SparseLinearizationd::SetInitialized,
           py::arg("initialized") = true)
      .def("error", &sym::SparseLinearizationd::Error)
      .def("linear_error", &sym::SparseLinearizationd::LinearError, py::arg("x_update"))
      .def(py::pickle(
          [](const sym::SparseLinearizationd& linearization) {  //  __getstate__
            return py::make_tuple(linearization.residual, linearization.hessian_lower,
                                  linearization.jacobian, linearization.rhs,
                                  linearization.IsInitialized());
          },
          [](py::tuple state) {  // __setstate__
            if (state.size() != 5) {
              throw py::value_error("Linearization.__setstate__ expected tuple of size 5.");
            }
            sym::SparseLinearizationd linearization;
            linearization.residual = state[0].cast<SparseLinearizationd::Vector>();
            linearization.hessian_lower = state[1].cast<SparseLinearizationd::Matrix>();
            linearization.jacobian = state[2].cast<SparseLinearizationd::Matrix>();
            linearization.rhs = state[3].cast<SparseLinearizationd::Vector>();
            linearization.SetInitialized(state[4].cast<bool>());
            return linearization;
          }));
}

}  // namespace sym
