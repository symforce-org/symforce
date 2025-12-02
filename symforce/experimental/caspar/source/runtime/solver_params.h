/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

namespace caspar {

struct SolverParams {
  float diag_init = 1.0;
  float diag_scaling_up = 2.0;
  float diag_scaling_down = 0.333333f;
  float diag_exit_value = 1e3;
  float diag_min = 1e-12;
  float score_exit_value = 0.0f;
  int solver_iter_max = 100;
  int pcg_iter_max = 20;
  float pcg_rel_error_exit = 1e-4;
  float pcg_rel_score_exit = -1.0f;    // disabled if == -1.0f
  float pcg_rel_decrease_min = -1.0f;  // disabled if == -1.0f
  float solver_rel_decrease_min = 1.0f;
};

}  // namespace caspar
