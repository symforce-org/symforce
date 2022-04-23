/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./optimizer.h"

sym::optimizer_params_t sym::DefaultOptimizerParams() {
  const bool verbose = true;
  const double initial_lambda = 1.0;
  const double lambda_up_factor = 4.0;
  const double lambda_down_factor = 1 / 4.0;
  const double lambda_lower_bound = 0.0;
  const double lambda_upper_bound = 1000000.0;
  const bool use_diagonal_damping = false;
  const bool use_unit_damping = true;
  const bool keep_max_diagonal_damping = false;
  const double diagonal_damping_min = 1e-6;
  const int iterations = 50;
  const double early_exit_min_reduction = 1e-6;
  const bool enable_bold_updates = false;

  return sym::optimizer_params_t{
      verbose,
      initial_lambda,
      lambda_up_factor,
      lambda_down_factor,
      lambda_lower_bound,
      lambda_upper_bound,
      use_diagonal_damping,
      use_unit_damping,
      keep_max_diagonal_damping,
      diagonal_damping_min,
      iterations,
      early_exit_min_reduction,
      enable_bold_updates,
  };
}

// Explicitly instantiate most commonly used optimizer templates to allow for faster compilation
// times.
template class sym::Optimizer<double>;
template class sym::Optimizer<float>;
