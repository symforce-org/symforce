/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./internal/covariance_utils.h"
#include "./internal/derivative_checker.h"
#include "./optimizer.h"

namespace sym {

// ----------------------------------------------------------------------------
// Constructors
// ----------------------------------------------------------------------------

template <typename ScalarType, typename NonlinearSolverType>
Optimizer<ScalarType, NonlinearSolverType>::Optimizer(const optimizer_params_t& params,
                                                      const std::vector<Factor<Scalar>>& factors,
                                                      const Scalar epsilon, const std::string& name,
                                                      const std::vector<Key>& keys,
                                                      bool debug_stats, bool check_derivatives)
    : factors_(factors),
      name_(name),
      nonlinear_solver_(params, name, epsilon),
      epsilon_(epsilon),
      debug_stats_(debug_stats),
      keys_(keys.empty() ? ComputeKeysToOptimize(factors_) : keys),
      index_(),
      linearizer_(factors_, keys_),
      linearize_func_(BuildLinearizeFunc(check_derivatives)) {}

template <typename ScalarType, typename NonlinearSolverType>
template <typename... NonlinearSolverArgs>
Optimizer<ScalarType, NonlinearSolverType>::Optimizer(
    const optimizer_params_t& params, const std::vector<Factor<Scalar>>& factors,
    const Scalar epsilon, const std::string& name, const std::vector<Key>& keys, bool debug_stats,
    bool check_derivatives, NonlinearSolverArgs&&... nonlinear_solver_args)
    : factors_(factors),
      name_(name),
      nonlinear_solver_(params, name, epsilon,
                        std::forward<NonlinearSolverArgs>(nonlinear_solver_args)...),
      epsilon_(epsilon),
      debug_stats_(debug_stats),
      keys_(keys.empty() ? ComputeKeysToOptimize(factors_) : keys),
      index_(),
      linearizer_(factors_, keys_),
      linearize_func_(BuildLinearizeFunc(check_derivatives)) {}

template <typename ScalarType, typename NonlinearSolverType>
Optimizer<ScalarType, NonlinearSolverType>::Optimizer(const optimizer_params_t& params,
                                                      std::vector<Factor<Scalar>>&& factors,
                                                      const Scalar epsilon, const std::string& name,
                                                      std::vector<Key>&& keys, bool debug_stats,
                                                      bool check_derivatives)
    : factors_(std::move(factors)),
      name_(name),
      nonlinear_solver_(params, name, epsilon),
      epsilon_(epsilon),
      debug_stats_(debug_stats),
      keys_(keys.empty() ? ComputeKeysToOptimize(factors_) : std::move(keys)),
      index_(),
      linearizer_(factors_, keys_),
      linearize_func_(BuildLinearizeFunc(check_derivatives)) {}

template <typename ScalarType, typename NonlinearSolverType>
template <typename... NonlinearSolverArgs>
Optimizer<ScalarType, NonlinearSolverType>::Optimizer(
    const optimizer_params_t& params, std::vector<Factor<Scalar>>&& factors, const Scalar epsilon,
    const std::string& name, std::vector<Key>&& keys, bool debug_stats, bool check_derivatives,
    NonlinearSolverArgs&&... nonlinear_solver_args)
    : factors_(std::move(factors)),
      nonlinear_solver_(params, name, epsilon,
                        std::forward<NonlinearSolverArgs>(nonlinear_solver_args)...),
      epsilon_(epsilon),
      debug_stats_(debug_stats),
      keys_(keys.empty() ? ComputeKeysToOptimize(factors_) : std::move(keys)),
      index_(),
      linearizer_(factors_, keys_),
      linearize_func_(BuildLinearizeFunc(check_derivatives)) {}

// ----------------------------------------------------------------------------
// Public methods
// ----------------------------------------------------------------------------

template <typename ScalarType, typename NonlinearSolverType>
OptimizationStats<ScalarType> Optimizer<ScalarType, NonlinearSolverType>::Optimize(
    Values<Scalar>* const values, int num_iterations, bool populate_best_linearization) {
  OptimizationStats<Scalar> stats{};
  Optimize(values, num_iterations, populate_best_linearization, &stats);
  return stats;
}

template <typename ScalarType, typename NonlinearSolverType>
void Optimizer<ScalarType, NonlinearSolverType>::Optimize(Values<Scalar>* const values,
                                                          int num_iterations,
                                                          bool populate_best_linearization,
                                                          OptimizationStats<Scalar>* const stats) {
  SYM_TIME_SCOPE("Optimizer<{}>::Optimize", name_);
  SYM_ASSERT(values != nullptr);
  SYM_ASSERT(stats != nullptr);

  if (num_iterations < 0) {
    num_iterations = nonlinear_solver_.Params().iterations;
  }

  stats->iterations.reserve(num_iterations);

  Initialize(*values);

  // Clear state for this run
  nonlinear_solver_.Reset(*values);
  stats->iterations.clear();
  IterateToConvergence(values, num_iterations, populate_best_linearization, stats);
}

template <typename ScalarType, typename NonlinearSolverType>
void Optimizer<ScalarType, NonlinearSolverType>::Optimize(Values<Scalar>* values,
                                                          int num_iterations,
                                                          OptimizationStats<Scalar>* stats) {
  return Optimize(values, num_iterations, false, stats);
}

template <typename ScalarType, typename NonlinearSolverType>
void Optimizer<ScalarType, NonlinearSolverType>::Optimize(Values<Scalar>* values,
                                                          OptimizationStats<Scalar>* stats) {
  return Optimize(values, -1, false, stats);
}

template <typename ScalarType, typename NonlinearSolverType>
Linearization<ScalarType> Optimizer<ScalarType, NonlinearSolverType>::Linearize(
    const Values<Scalar>& values) {
  Initialize(values);

  Linearization<Scalar> linearization;
  linearize_func_(values, &linearization);
  return linearization;
}

template <typename ScalarType, typename NonlinearSolverType>
void Optimizer<ScalarType, NonlinearSolverType>::ComputeAllCovariances(
    const Linearization<Scalar>& linearization,
    std::unordered_map<Key, MatrixX<Scalar>>* const covariances_by_key) {
  SYM_ASSERT(IsInitialized());
  nonlinear_solver_.ComputeCovariance(linearization.hessian_lower,
                                      &compute_covariances_storage_.covariance);
  linearizer_.SplitCovariancesByKey(compute_covariances_storage_.covariance, keys_,
                                    covariances_by_key);
}

template <typename ScalarType, typename NonlinearSolverType>
void Optimizer<ScalarType, NonlinearSolverType>::ComputeCovariances(
    const Linearization<Scalar>& linearization, const std::vector<Key>& keys,
    std::unordered_map<Key, MatrixX<Scalar>>* const covariances_by_key) {
  size_t block_dim{};
  const bool contiguous = linearizer_.CheckKeysAreContiguousAtStart(keys, &block_dim);
  SYM_ASSERT(contiguous);

  // Copy into modifiable storage
  compute_covariances_storage_.H_damped = linearization.hessian_lower;

  internal::ComputeCovarianceBlockWithSchurComplement(&compute_covariances_storage_.H_damped,
                                                      block_dim, epsilon_,
                                                      &compute_covariances_storage_.covariance);
  linearizer_.SplitCovariancesByKey(compute_covariances_storage_.covariance, keys,
                                    covariances_by_key);
}

template <typename ScalarType, typename NonlinearSolverType>
const std::vector<Key>& Optimizer<ScalarType, NonlinearSolverType>::Keys() const {
  return keys_;
}

template <typename ScalarType, typename NonlinearSolverType>
const std::vector<Factor<ScalarType>>& Optimizer<ScalarType, NonlinearSolverType>::Factors() const {
  return factors_;
}

template <typename ScalarType, typename NonlinearSolverType>
const Linearizer<ScalarType>& Optimizer<ScalarType, NonlinearSolverType>::Linearizer() const {
  return linearizer_;
}

template <typename ScalarType, typename NonlinearSolverType>
Linearizer<ScalarType>& Optimizer<ScalarType, NonlinearSolverType>::Linearizer() {
  return linearizer_;
}

template <typename ScalarType, typename NonlinearSolverType>
void Optimizer<ScalarType, NonlinearSolverType>::UpdateParams(const optimizer_params_t& params) {
  nonlinear_solver_.UpdateParams(params);
}

// ----------------------------------------------------------------------------
// Protected methods
// ----------------------------------------------------------------------------

template <typename ScalarType, typename NonlinearSolverType>
void Optimizer<ScalarType, NonlinearSolverType>::IterateToConvergence(
    Values<Scalar>* const values, const size_t num_iterations,
    const bool populate_best_linearization, OptimizationStats<Scalar>* const stats) {
  SYM_TIME_SCOPE("Optimizer<{}>::IterateToConvergence", name_);
  bool optimization_early_exited = false;

  // Iterate
  for (int i = 0; i < num_iterations; i++) {
    const bool should_early_exit = nonlinear_solver_.Iterate(linearize_func_, stats, debug_stats_);
    if (should_early_exit) {
      optimization_early_exited = true;
      break;
    }
  }

  {
    SYM_TIME_SCOPE("Optimizer<{}>::CopyValuesAndLinearization", name_);
    // Save best results
    (*values) = nonlinear_solver_.GetBestValues();

    if (populate_best_linearization) {
      // NOTE(aaron): This makes a copy, which doesn't seem ideal.  We could instead put a
      // Linearization** in the stats, but then we'd have the issue of defining when the pointer
      // becomes invalid
      stats->best_linearization = nonlinear_solver_.GetBestLinearization();
    }
  }

  stats->early_exited = optimization_early_exited;
}

template <typename ScalarType, typename NonlinearSolverType>
bool Optimizer<ScalarType, NonlinearSolverType>::IsInitialized() const {
  return index_.entries.size() != 0;
}

template <typename ScalarType, typename NonlinearSolverType>
typename NonlinearSolverType::LinearizeFunc
Optimizer<ScalarType, NonlinearSolverType>::BuildLinearizeFunc(const bool check_derivatives) {
  return [this, check_derivatives](const Values<Scalar>& values,
                                   Linearization<Scalar>* const linearization) {
    linearizer_.Relinearize(values, linearization);

    if (check_derivatives) {
      SYM_ASSERT(linearization != nullptr);
      SYM_ASSERT(
          internal::CheckDerivatives(&linearizer_, values, index_, *linearization, epsilon_));
    }
  };
}

template <typename ScalarType, typename NonlinearSolverType>
void Optimizer<ScalarType, NonlinearSolverType>::Initialize(const Values<Scalar>& values) {
  if (!IsInitialized()) {
    index_ = values.CreateIndex(keys_);
    nonlinear_solver_.SetIndex(index_);
  }
}

template <typename ScalarType, typename NonlinearSolverType>
const std::string& Optimizer<ScalarType, NonlinearSolverType>::GetName() {
  return name_;
}

extern template class Optimizer<double>;
extern template class Optimizer<float>;

}  // namespace sym
