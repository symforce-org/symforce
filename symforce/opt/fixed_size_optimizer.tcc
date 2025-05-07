/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include "./assert.h"
#include "./fixed_size_optimizer.h"
#include "./optimizer.h"

namespace sym {

// ----------------------------------------------------------------------------
// Constructors
// ----------------------------------------------------------------------------

template <typename ScalarType, typename NonlinearSolverType>
FixedSizeOptimizer<ScalarType, NonlinearSolverType>::FixedSizeOptimizer(
    const optimizer_params_t& params, const std::string& name, const Scalar epsilon)
    : name_(name),
      nonlinear_solver_(params, name, epsilon),
      epsilon_(epsilon),
      debug_stats_(params.debug_stats),
      verbose_(params.verbose) {
  SYM_ASSERT(!params.check_derivatives || params.include_jacobians);
}

template <typename ScalarType, typename NonlinearSolverType>
template <typename... NonlinearSolverArgs>
FixedSizeOptimizer<ScalarType, NonlinearSolverType>::FixedSizeOptimizer(
    const optimizer_params_t& params, const std::string& name, const Scalar epsilon,
    NonlinearSolverArgs&&... nonlinear_solver_args)
    : name_(name),
      nonlinear_solver_(params, name, epsilon,
                        std::forward<NonlinearSolverArgs>(nonlinear_solver_args)...),
      epsilon_(epsilon),
      debug_stats_(params.debug_stats),
      verbose_(params.verbose) {
  SYM_ASSERT(!params.check_derivatives || params.include_jacobians);
}

// ----------------------------------------------------------------------------
// Public methods
// ----------------------------------------------------------------------------

template <typename ScalarType, typename NonlinearSolverType>
typename FixedSizeOptimizer<ScalarType, NonlinearSolverType>::Stats
FixedSizeOptimizer<ScalarType, NonlinearSolverType>::Optimize(
    ValuesType& values, const LinearizeFunc& linearize_func, const int num_iterations,
    const bool populate_best_linearization) {
  Stats stats{};
  Optimize(values, linearize_func, num_iterations, populate_best_linearization, stats);
  return stats;
}

template <typename ScalarType, typename NonlinearSolverType>
void FixedSizeOptimizer<ScalarType, NonlinearSolverType>::Optimize(
    ValuesType& values, const LinearizeFunc& linearize_func, int num_iterations,
    bool populate_best_linearization, Stats& stats) {
  OptimizeImpl(values, nonlinear_solver_, linearize_func, num_iterations,
               populate_best_linearization, name_, verbose_, stats);
}

template <typename ScalarType, typename NonlinearSolverType>
void FixedSizeOptimizer<ScalarType, NonlinearSolverType>::Optimize(
    ValuesType& values, const LinearizeFunc& linearize_func, int num_iterations, Stats& stats) {
  return Optimize(values, linearize_func, num_iterations, false, stats);
}

template <typename ScalarType, typename NonlinearSolverType>
void FixedSizeOptimizer<ScalarType, NonlinearSolverType>::Optimize(
    ValuesType& values, const LinearizeFunc& linearize_func, Stats& stats) {
  return Optimize(values, linearize_func, -1, false, stats);
}

template <typename ScalarType, typename NonlinearSolverType>
void FixedSizeOptimizer<ScalarType, NonlinearSolverType>::ComputeFullCovariance(
    const Linearization<MatrixType>& linearization, MatrixX<Scalar>& covariance) {
  nonlinear_solver_.ComputeCovariance(linearization.hessian_lower, covariance);
}

template <typename ScalarType, typename NonlinearSolverType>
const NonlinearSolverType& FixedSizeOptimizer<ScalarType, NonlinearSolverType>::NonlinearSolver()
    const {
  return nonlinear_solver_;
}

template <typename ScalarType, typename NonlinearSolverType>
NonlinearSolverType& FixedSizeOptimizer<ScalarType, NonlinearSolverType>::NonlinearSolver() {
  return nonlinear_solver_;
}

template <typename ScalarType, typename NonlinearSolverType>
void FixedSizeOptimizer<ScalarType, NonlinearSolverType>::UpdateParams(
    const optimizer_params_t& params) {
  nonlinear_solver_.UpdateParams(params);
}

template <typename ScalarType, typename NonlinearSolverType>
const optimizer_params_t& FixedSizeOptimizer<ScalarType, NonlinearSolverType>::Params() const {
  return nonlinear_solver_.Params();
}

extern template class FixedSizeOptimizer<double>;
extern template class FixedSizeOptimizer<float>;

}  // namespace sym
