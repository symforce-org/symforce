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
                                                      const std::string& name,
                                                      std::vector<Key> keys, const Scalar epsilon)
    : factors_(std::move(factors)),
      name_(name),
      nonlinear_solver_(params, name, epsilon),
      epsilon_(epsilon),
      debug_stats_(params.debug_stats),
      include_jacobians_(params.include_jacobians),
      keys_(keys.empty() ? ComputeKeysToOptimize(factors_) : std::move(keys)),
      index_(),
      linearizer_(name_, factors_, keys_, params.include_jacobians, params.debug_checks),
      linearize_func_(BuildLinearizeFunc(params.check_derivatives)),
      verbose_(params.verbose) {
  SYM_ASSERT(factors_.size() > 0);
  SYM_ASSERT(keys_.size() > 0);

  SYM_ASSERT(!params.check_derivatives || params.include_jacobians);
}

template <typename ScalarType, typename NonlinearSolverType>
Optimizer<ScalarType, NonlinearSolverType>::Optimizer(const optimizer_params_t& params,
                                                      std::vector<Factor<Scalar>> factors,
                                                      const Scalar epsilon, const std::string& name,
                                                      std::vector<Key> keys, const bool debug_stats,
                                                      const bool check_derivatives,
                                                      const bool include_jacobians)
    : Optimizer(
          [&]() {
            auto modified_params = params;
            modified_params.debug_stats = debug_stats;
            modified_params.check_derivatives = check_derivatives;
            modified_params.include_jacobians = include_jacobians;
            return modified_params;
          }(),
          std::move(factors), name, std::move(keys), epsilon) {}

template <typename ScalarType, typename NonlinearSolverType>
template <typename... NonlinearSolverArgs>
Optimizer<ScalarType, NonlinearSolverType>::Optimizer(
    const optimizer_params_t& params, std::vector<Factor<Scalar>> factors, const std::string& name,
    std::vector<Key> keys, Scalar epsilon, NonlinearSolverArgs&&... nonlinear_solver_args)
    : factors_(std::move(factors)),
      name_(name),
      nonlinear_solver_(params, name, epsilon,
                        std::forward<NonlinearSolverArgs>(nonlinear_solver_args)...),
      epsilon_(epsilon),
      debug_stats_(params.debug_stats),
      include_jacobians_(params.include_jacobians),
      keys_(keys.empty() ? ComputeKeysToOptimize(factors_) : std::move(keys)),
      index_(),
      linearizer_(name_, factors_, keys_, params.include_jacobians),
      linearize_func_(BuildLinearizeFunc(params.check_derivatives)),
      verbose_(params.verbose) {
  SYM_ASSERT(factors_.size() > 0);
  SYM_ASSERT(keys_.size() > 0);

  SYM_ASSERT(!params.check_derivatives || params.include_jacobians);
}

template <typename ScalarType, typename NonlinearSolverType>
template <typename... NonlinearSolverArgs>
Optimizer<ScalarType, NonlinearSolverType>::Optimizer(
    const optimizer_params_t& params, std::vector<Factor<Scalar>> factors, const Scalar epsilon,
    const std::string& name, std::vector<Key> keys, bool debug_stats, bool check_derivatives,
    bool include_jacobians, NonlinearSolverArgs&&... nonlinear_solver_args)
    : Optimizer(
          [&]() {
            auto modified_params = params;
            modified_params.debug_stats = debug_stats;
            modified_params.check_derivatives = check_derivatives;
            modified_params.include_jacobians = include_jacobians;
            return modified_params;
          }(),
          std::move(factors), name, std::move(keys), epsilon,
          std::forward<NonlinearSolverArgs>(nonlinear_solver_args)...) {}

// ----------------------------------------------------------------------------
// Public methods
// ----------------------------------------------------------------------------

template <typename ScalarType, typename NonlinearSolverType>
typename sym::Optimizer<ScalarType, NonlinearSolverType>::Stats
Optimizer<ScalarType, NonlinearSolverType>::Optimize(Values<Scalar>& values, int num_iterations,
                                                     bool populate_best_linearization) {
  Stats stats{};
  Optimize(values, num_iterations, populate_best_linearization, stats);
  return stats;
}

template <typename ScalarType, typename NonlinearSolverType>
void Optimizer<ScalarType, NonlinearSolverType>::Optimize(Values<Scalar>& values,
                                                          int num_iterations,
                                                          bool populate_best_linearization,
                                                          Stats& stats) {
  SYM_TIME_SCOPE("Optimizer<{}>::Optimize", name_);

  if (num_iterations < 0) {
    num_iterations = nonlinear_solver_.Params().iterations;
  }

  Initialize(values);

  // Clear state for this run
  nonlinear_solver_.Reset(values);
  stats.Reset(num_iterations);
  IterateToConvergence(values, num_iterations, populate_best_linearization, stats);

  if (verbose_) {
    if (stats.status == optimization_status_t::FAILED) {
      spdlog::warn("LM<{}> Optimization finished with status: FAILED, reason: {}", name_,
                   FailureReason::from_int(stats.failure_reason));
    } else {
      spdlog::log(stats.status == optimization_status_t::SUCCESS ? spdlog::level::info
                                                                 : spdlog::level::warn,
                  "LM<{}> Optimization finished with status: {}", name_, stats.status);
    }
  }
}

template <typename ScalarType, typename NonlinearSolverType>
void Optimizer<ScalarType, NonlinearSolverType>::Optimize(Values<Scalar>& values,
                                                          int num_iterations, Stats& stats) {
  return Optimize(values, num_iterations, false, stats);
}

template <typename ScalarType, typename NonlinearSolverType>
void Optimizer<ScalarType, NonlinearSolverType>::Optimize(Values<Scalar>& values, Stats& stats) {
  return Optimize(values, -1, false, stats);
}

template <typename ScalarType, typename NonlinearSolverType>
Linearization<typename NonlinearSolverType::MatrixType>
Optimizer<ScalarType, NonlinearSolverType>::Linearize(const Values<Scalar>& values) {
  Initialize(values);

  Linearization<MatrixType> linearization;
  linearize_func_(values, linearization);
  return linearization;
}

template <typename ScalarType, typename NonlinearSolverType>
void Optimizer<ScalarType, NonlinearSolverType>::ComputeAllCovariances(
    const Linearization<MatrixType>& linearization,
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
template <typename LinearizerType>
bool CheckKeyOrderMatchesLinearizerKeysStart(const LinearizerType& linearizer,
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
template <typename LinearizerType>
size_t ComputeBlockDimension(const LinearizerType& linearizer, const std::vector<Key>& keys) {
  // The idea is that the offset of a state index entry is the sum of the tangent dims of all of
  // the previous keys, so we just add the tangent_dim of the last key to it's offset to get the
  // sum of all tangent dims.
  const auto& last_index_entry = linearizer.StateIndex().at(keys.back().GetLcmType());
  return last_index_entry.offset + last_index_entry.tangent_dim;
}

}  // namespace internal

template <typename ScalarType, typename NonlinearSolverType>
void Optimizer<ScalarType, NonlinearSolverType>::ComputeCovariances(
    const Linearization<MatrixType>& linearization, const std::vector<Key>& keys,
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
void Optimizer<ScalarType, NonlinearSolverType>::ComputeFullCovariance(
    const Linearization<MatrixType>& linearization, MatrixX<Scalar>& covariance) {
  SYM_ASSERT(IsInitialized());
  nonlinear_solver_.ComputeCovariance(linearization.hessian_lower, covariance);
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
const typename Optimizer<ScalarType, NonlinearSolverType>::LinearizerType&
Optimizer<ScalarType, NonlinearSolverType>::Linearizer() const {
  return linearizer_;
}

template <typename ScalarType, typename NonlinearSolverType>
typename Optimizer<ScalarType, NonlinearSolverType>::LinearizerType&
Optimizer<ScalarType, NonlinearSolverType>::Linearizer() {
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
    Stats& stats) {
  SYM_TIME_SCOPE("Optimizer<{}>::IterateToConvergence", name_);
  SYM_ASSERT(num_iterations > 0, "num_iterations must be positive, got {}", num_iterations);

  // Iterate
  int i;
  for (i = 0; i < num_iterations; i++) {
    const auto maybe_status_and_failure_reason = nonlinear_solver_.Iterate(linearize_func_, stats);
    if (maybe_status_and_failure_reason) {
      const auto& status_and_failure_reason = maybe_status_and_failure_reason.value();

      SYM_ASSERT(status_and_failure_reason.first != optimization_status_t::INVALID,
                 "NonlinearSolver::Iterate should never return INVALID");
      SYM_ASSERT(status_and_failure_reason.first != optimization_status_t::HIT_ITERATION_LIMIT,
                 "NonlinearSolver::Iterate should never return HIT_ITERATION_LIMIT");

      stats.status = status_and_failure_reason.first;
      stats.failure_reason = status_and_failure_reason.second.int_value();
      break;
    }
  }

  if (i == num_iterations) {
    stats.status = optimization_status_t::HIT_ITERATION_LIMIT;
    stats.failure_reason = {};
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

  if (debug_stats_ && include_jacobians_) {
    const auto& linearization = nonlinear_solver_.GetBestLinearization();
    stats.jacobian_sparsity = GetSparseStructure(linearization.jacobian);
  }
}

template <typename ScalarType, typename NonlinearSolverType>
bool Optimizer<ScalarType, NonlinearSolverType>::IsInitialized() const {
  return index_.entries.size() != 0;
}

template <typename ScalarType, typename NonlinearSolverType>
typename NonlinearSolverType::LinearizeFunc
Optimizer<ScalarType, NonlinearSolverType>::BuildLinearizeFunc(const bool check_derivatives) {
  return [this, check_derivatives](const Values<Scalar>& values,
                                   Linearization<MatrixType>& linearization) {
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
