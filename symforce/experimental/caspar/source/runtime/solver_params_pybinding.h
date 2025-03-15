/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <pybind11/pybind11.h>

#include "solver_params.h"

namespace py = pybind11;

namespace caspar {

void add_solver_params_pybinding(py::module_ module) {
  py::class_<SolverParams>(module, "SolverParams", "Class for setting solver parameters.")
      .def(py::init<>())
      .def_readwrite("solver_iter_max", &SolverParams::solver_iter_max)
      .def_readwrite("pcg_iter_max", &SolverParams::pcg_iter_max)
      .def_readwrite("diag_init", &SolverParams::diag_init)
      .def_readwrite("diag_scaling_up", &SolverParams::diag_scaling_up)
      .def_readwrite("diag_scaling_down", &SolverParams::diag_scaling_down)
      .def_readwrite("diag_exit_value", &SolverParams::diag_exit_value)
      .def_readwrite("diag_min", &SolverParams::diag_min)
      .def_readwrite("solver_rel_decrease_min", &SolverParams::solver_rel_decrease_min)
      .def_readwrite("score_exit_value", &SolverParams::diag_exit_value)
      .def_readwrite("pcg_rel_decrease_min", &SolverParams::pcg_rel_decrease_min)
      .def_readwrite("pcg_rel_error_exit", &SolverParams::pcg_rel_error_exit)
      .def_readwrite("pcg_rel_score_exit", &SolverParams::pcg_rel_score_exit);
}

}  // namespace caspar
