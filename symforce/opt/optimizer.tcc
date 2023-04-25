/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include "./assert.h"
#include "./internal/covariance_utils.h"
#include "./internal/derivative_checker.h"
#include "./optimizer.h"

namespace sym {

// ----------------------------------------------------------------------------
// Constructors
// ----------------------------------------------------------------------------

template <typename ScalarType, typename NonlinearSolverType>
Optimizer<ScalarType, NonlinearSolverType>::Optimizer(const optimizer_params_t& params,
                                                      std::vector<Factor<Scalar>> factors,
                                                      const Scalar epsilon, const std::string& name,
                                                      std::vector<Key> keys, bool debug_stats,
                                                      bool check_derivatives,
                                                      bool include_jacobians)
    : factors_(std::move(factors)),
      name_(name),
      nonlinear_solver_(params, name, epsilon),
      epsilon_(epsilon),
      debug_stats_(debug_stats),
      include_jacobians_(include_jacobians),
      keys_(keys.empty() ? ComputeKeysToOptimize(factors_) : std::move(keys)),
      index_(),
      linearizer_(name_, factors_, keys_, include_jacobians),
      linearize_func_(BuildLinearizeFunc(check_derivatives)) {
  SYM_ASSERT(factors_.size() > 0);
  SYM_ASSERT(keys_.size() > 0);

  SYM_ASSERT(!check_derivatives || include_jacobians);
}

template <typename ScalarType, typename NonlinearSolverType>
template <typename... NonlinearSolverArgs>
Optimizer<ScalarType, NonlinearSolverType>::Optimizer(
    const optimizer_params_t& params, std::vector<Factor<Scalar>> factors, const Scalar epsilon,
    const std::string& name, std::vector<Key> keys, bool debug_stats, bool check_derivatives,
    bool include_jacobians, NonlinearSolverArgs&&... nonlinear_solver_args)
    : factors_(std::move(factors)),
      name_(name),
      nonlinear_solver_(params, name, epsilon,
                        std::forward<NonlinearSolverArgs>(nonlinear_solver_args)...),
      epsilon_(epsilon),
      debug_stats_(debug_stats),
      include_jacobians_(include_jacobians),
      keys_(keys.empty() ? ComputeKeysToOptimize(factors_) : std::move(keys)),
      index_(),
      linearizer_(name_, factors_, keys_, include_jacobians),
      linearize_func_(BuildLinearizeFunc(check_derivatives)) {
  SYM_ASSERT(factors_.size() > 0);
  SYM_ASSERT(keys_.size() > 0);

  SYM_ASSERT(!check_derivatives || include_jacobians);
}

// ----------------------------------------------------------------------------
// Public methods
// ----------------------------------------------------------------------------

template <typename ScalarType, typename NonlinearSolverType>
OptimizationStats<ScalarType> Optimizer<ScalarType, NonlinearSolverType>::Optimize(
    Values<Scalar>& values, int num_iterations, bool populate_best_linearization) {
  OptimizationStats<Scalar> stats{};
  Optimize(values, num_iterations, populate_best_linearization, stats);
  return stats;
}

template <typename ScalarType, typename NonlinearSolverType>
OptimizationStats<ScalarType> Optimizer<ScalarType, NonlinearSolverType>::Optimize(
    Values<Scalar>* values, int num_iterations, bool populate_best_linearization) {
  SYM_ASSERT(values != nullptr);
  return Optimize(*values, num_iterations, populate_best_linearization);
}

template <typename ScalarType, typename NonlinearSolverType>
void Optimizer<ScalarType, NonlinearSolverType>::Optimize(Values<Scalar>& values,
                                                          int num_iterations,
                                                          bool populate_best_linearization,
                                                          OptimizationStats<Scalar>& stats) {
  SYM_TIME_SCOPE("Optimizer<{}>::Optimize", name_);

  if (num_iterations < 0) {
    num_iterations = nonlinear_solver_.Params().iterations;
  }

  Initialize(values);

  // Clear state for this run
  nonlinear_solver_.Reset(values);
  stats.Reset(num_iterations);
  IterateToConvergence(values, num_iterations, populate_best_linearization, stats);
}

template <typename ScalarType, typename NonlinearSolverType>
void Optimizer<ScalarType, NonlinearSolverType>::Optimize(Values<Scalar>* values,
                                                          int num_iterations,
                                                          bool populate_best_linearization,
                                                          OptimizationStats<Scalar>* stats) {
  SYM_ASSERT(values != nullptr);
  SYM_ASSERT(stats != nullptr);
  Optimize(*values, num_iterations, populate_best_linearization, *stats);
}

template <typename ScalarType, typename NonlinearSolverType>
void Optimizer<ScalarType, NonlinearSolverType>::Optimize(Values<Scalar>& values,
                                                          int num_iterations,
                                                          OptimizationStats<Scalar>& stats) {
  return Optimize(values, num_iterations, false, stats);
}

template <typename ScalarType, typename NonlinearSolverType>
void Optimizer<ScalarType, NonlinearSolverType>::Optimize(Values<Scalar>* values,
                                                          int num_iterations,
                                                          OptimizationStats<Scalar>* stats) {
  SYM_ASSERT(values != nullptr);
  SYM_ASSERT(stats != nullptr);
  Optimize(*values, num_iterations, *stats);
}

template <typename ScalarType, typename NonlinearSolverType>
void Optimizer<ScalarType, NonlinearSolverType>::Optimize(Values<Scalar>& values,
                                                          OptimizationStats<Scalar>& stats) {
  return Optimize(values, -1, false, stats);
}

template <typename ScalarType, typename NonlinearSolverType>
void Optimizer<ScalarType, NonlinearSolverType>::Optimize(Values<Scalar>* values,
                                                          OptimizationStats<Scalar>* stats) {
  SYM_ASSERT(values != nullptr);
  SYM_ASSERT(stats != nullptr);
  Optimize(*values, *stats);
}

template <typename ScalarType, typename NonlinearSolverType>
SparseLinearization<ScalarType> Optimizer<ScalarType, NonlinearSolverType>::Linearize(
    const Values<Scalar>& values) {
  Initialize(values);

  SparseLinearization<Scalar> linearization;
  linearize_func_(values, linearization);
  return linearization;
}

template <typename ScalarType, typename NonlinearSolverType>
void Optimizer<ScalarType, NonlinearSolverType>::ComputeAllCovariances(
    const SparseLinearization<Scalar>& linearization,
    std::unordered_map<Key, MatrixX<Scalar>>& covariances_by_key) {
  SYM_ASSERT(IsInitialized());
  nonlinear_solver_.ComputeCovariance(linearization.hessian_lower,
                                      compute_covariances_storage_.covariance);
  internal::SplitCovariancesByKey(linearizer_, compute_covariances_storage_.covariance, keys_,
                                  covariances_by_key);
}

namespace internal {

/**
 * Check whether the keys in `keys` correspond 1-1 (and in the same order) with the start of the
 * key ordering in the problem linearization
 *
 * Throws runtime_error if keys is longer than linearizer.Keys() or sometimes if key not in Keys
 * found at start of linearizer.Keys().
 *
 * TODO(aaron): Maybe kill this once we have general marginalization
 */
template <typename Scalar>
bool CheckKeyOrderMatchesLinearizerKeysStart(const Linearizer<Scalar>& linearizer,
                                             const std::vector<Key>& keys) {
  SYM_ASSERT(!keys.empty());

  const std::vector<Key>& full_problem_keys = linearizer.Keys();
  if (full_problem_keys.size() < keys.size()) {
    throw std::runtime_error("Keys has extra entries that are not in the full problem");
  }

  const std::unordered_map<key_t, index_entry_t>& state_index = linearizer.StateIndex();
  for (int i = 0; i < static_cast<int>(keys.size()); i++) {
    if (keys[i] != full_problem_keys[i]) {
      if (state_index.find(keys[i].GetLcmType()) == state_index.end()) {
        throw std::runtime_error("Tried to check key which is not in the full problem");
      } else {
        // The next key is in the problem, it's just out of order; so we return false
        return false;
      }
    }
  }

  return true;
}

/**
 * Returns the (tangent) dimension of the problem hessian and rhs which is occupied by
 * the given keys.
 *
 * Precondition: keys equals first keys.size() entries of linearizer.Keys()
 */
template <typename Scalar>
size_t ComputeBlockDimension(const Linearizer<Scalar>& linearizer, const std::vector<Key>& keys) {
  // The idea is that the offset of a state index entry is the sum of the tangent dims of all of
  // the previous keys, so we just add the tangent_dim of the last key to it's offset to get the
  // sum of all tangent dims.
  const auto& last_index_entry = linearizer.StateIndex().at(keys.back().GetLcmType());
  return last_index_entry.offset + last_index_entry.tangent_dim;
}

}  // namespace internal

template <typename ScalarType, typename NonlinearSolverType>
void Optimizer<ScalarType, NonlinearSolverType>::ComputeAllCovariances(
    const SparseLinearization<Scalar>& linearization,
    std::unordered_map<Key, MatrixX<Scalar>>* covariances_by_key) {
  SYM_ASSERT(covariances_by_key != nullptr);
  ComputeAllCovariances(linearization, *covariances_by_key);
}

template <typename ScalarType, typename NonlinearSolverType>
void Optimizer<ScalarType, NonlinearSolverType>::ComputeCovariances(
    const SparseLinearization<Scalar>& linearization, const std::vector<Key>& keys,
    std::unordered_map<Key, MatrixX<Scalar>>& covariances_by_key) {
  const bool same_order = internal::CheckKeyOrderMatchesLinearizerKeysStart(linearizer_, keys);
  SYM_ASSERT(same_order);
  const size_t block_dim = internal::ComputeBlockDimension(linearizer_, keys);

  // Copy into modifiable storage
  compute_covariances_storage_.H_damped = linearization.hessian_lower;

  internal::ComputeCovarianceBlockWithSchurComplement(compute_covariances_storage_.H_damped,
                                                      block_dim, epsilon_,
                                                      compute_covariances_storage_.covariance);
  internal::SplitCovariancesByKey(linearizer_, compute_covariances_storage_.covariance, keys,
                                  covariances_by_key);
}

template <typename ScalarType, typename NonlinearSolverType>
void Optimizer<ScalarType, NonlinearSolverType>::ComputeCovariances(
    const SparseLinearization<Scalar>& linearization, const std::vector<Key>& keys,
    std::unordered_map<Key, MatrixX<Scalar>>* covariances_by_key) {
  SYM_ASSERT(covariances_by_key != nullptr);
  ComputeCovariances(linearization, keys, *covariances_by_key);
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

template <typename ScalarType, typename NonlinearSolverType>
const optimizer_params_t& Optimizer<ScalarType, NonlinearSolverType>::Params() const {
  return nonlinear_solver_.Params();
}

// ----------------------------------------------------------------------------
// Protected methods
// ----------------------------------------------------------------------------

template <typename ScalarType, typename NonlinearSolverType>
void Optimizer<ScalarType, NonlinearSolverType>::IterateToConvergence(
    Values<Scalar>& values, const int num_iterations, const bool populate_best_linearization,
    OptimizationStats<Scalar>& stats) {
  SYM_TIME_SCOPE("Optimizer<{}>::IterateToConvergence", name_);
  bool optimization_early_exited = false;

  // Iterate
  for (int i = 0; i < num_iterations; i++) {
    const bool should_early_exit =
        nonlinear_solver_.Iterate(linearize_func_, stats, debug_stats_, include_jacobians_);
    if (should_early_exit) {
      optimization_early_exited = true;
      break;
    }
  }

  {
    SYM_TIME_SCOPE("Optimizer<{}>::CopyValuesAndLinearization", name_);
    // Save best results
    values = nonlinear_solver_.GetBestValues();

    if (populate_best_linearization) {
      // NOTE(aaron): This makes a copy, which doesn't seem ideal.  We could instead put a
      // Linearization** in the stats, but then we'd have the issue of defining when the pointer
      // becomes invalid
      stats.best_linearization = nonlinear_solver_.GetBestLinearization();
    }
  }

  if (debug_stats_) {
    const auto& linearization = nonlinear_solver_.GetBestLinearization();
    stats.jacobian_sparsity = GetSparseStructure(linearization.jacobian);
  }

  stats.early_exited = optimization_early_exited;
}

template <typename ScalarType, typename NonlinearSolverType>
bool Optimizer<ScalarType, NonlinearSolverType>::IsInitialized() const {
  return index_.entries.size() != 0;
}

template <typename ScalarType, typename NonlinearSolverType>
typename NonlinearSolverType::LinearizeFunc
Optimizer<ScalarType, NonlinearSolverType>::BuildLinearizeFunc(const bool check_derivatives) {
  return [this, check_derivatives](const Values<Scalar>& values,
                                   SparseLinearization<Scalar>& linearization) {
    linearizer_.Relinearize(values, linearization);

    if (check_derivatives) {
      SYM_ASSERT(internal::CheckDerivatives(linearizer_, values, index_, linearization, epsilon_));
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
