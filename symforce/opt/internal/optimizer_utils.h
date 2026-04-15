/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <spdlog/spdlog.h>

#include "../fmt_compat.h"
#include "../optimization_stats.h"
#include "../tic_toc.h"

namespace sym {

/**
 * Log the exit status of the optimization
 */
template <typename OptimizationStats, typename NonlinearSolver>
void LogStatus(const std::string& name, const OptimizationStats& stats) {
  if (stats.status == optimization_status_t::FAILED) {
    spdlog::warn("LM<{}> Optimization finished with status: FAILED, reason: {}", name,
                 NonlinearSolver::FailureReason::from_int(stats.failure_reason));
  } else {
    spdlog::log(
        stats.status == optimization_status_t::SUCCESS ? spdlog::level::info : spdlog::level::warn,
        "LM<{}> Optimization finished with status: {}", name, stats.status);
  }
}

/**
 * Call nonlinear_solver.Iterate() on the given values (updating in place) until out of
 * iterations or converged
 */
template <typename ValuesType, typename NonlinearSolver, typename LinearizeFunc,
          typename OptimizationStats>
void IterateToConvergenceImpl(ValuesType& values, NonlinearSolver& nonlinear_solver,
                              const LinearizeFunc& linearize_func, const int num_iterations,
                              const bool populate_best_linearization, const std::string& name,
                              OptimizationStats& stats) {
  SYM_TIME_SCOPE("Optimizer<{}>::IterateToConvergence", name);
  SYM_ASSERT(num_iterations > 0, "num_iterations must be positive, got {}", num_iterations);

  // Iterate
  int i;
  for (i = 0; i < num_iterations; i++) {
    const auto maybe_status_and_failure_reason = nonlinear_solver.Iterate(linearize_func, stats);
    if (maybe_status_and_failure_reason) {
      const auto& [status, failure_reason] = maybe_status_and_failure_reason.value();

      SYM_ASSERT(status != optimization_status_t::INVALID,
                 "NonlinearSolver::Iterate should never return INVALID");
      SYM_ASSERT(status != optimization_status_t::HIT_ITERATION_LIMIT,
                 "NonlinearSolver::Iterate should never return HIT_ITERATION_LIMIT");

      stats.status = status;
      stats.failure_reason = failure_reason.int_value();
      break;
    }
  }

  if (i == num_iterations) {
    stats.status = optimization_status_t::HIT_ITERATION_LIMIT;
    stats.failure_reason = {};
  }

  {
    SYM_TIME_SCOPE("Optimizer<{}>::CopyValuesAndLinearization", name);
    // Save best results
    values = nonlinear_solver.GetBestValues();

    if (populate_best_linearization) {
      // NOTE(aaron): This makes a copy, which doesn't seem ideal.  We could instead put a
      // Linearization** in the stats, but then we'd have the issue of defining when the pointer
      // becomes invalid
      stats.best_linearization = nonlinear_solver.GetBestLinearization();
    }
  }
}

/**
 * Optimize the given values in-place
 */
template <typename ValuesType, typename NonlinearSolver, typename LinearizeFunc,
          typename OptimizationStats>
void OptimizeImpl(ValuesType& values, NonlinearSolver& nonlinear_solver,
                  const LinearizeFunc& linearize_func, int num_iterations,
                  const bool populate_best_linearization, const std::string& name,
                  const bool verbose, OptimizationStats& stats) {
  SYM_TIME_SCOPE("Optimizer<{}>::Optimize", name);

  if (num_iterations < 0) {
    num_iterations = nonlinear_solver.Params().iterations;
  }

  // Clear state for this run
  nonlinear_solver.Reset(values);
  stats.Reset(num_iterations);
  IterateToConvergenceImpl(values, nonlinear_solver, linearize_func, num_iterations,
                           populate_best_linearization, name, stats);

  if (verbose) {
    LogStatus<OptimizationStats, NonlinearSolver>(name, stats);
  }
}

}  // namespace sym
