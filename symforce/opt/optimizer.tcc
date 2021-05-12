#include "./internal/derivative_checker.h"
#include "./optimizer.h"

namespace sym {

// ----------------------------------------------------------------------------
// Constructors
// ----------------------------------------------------------------------------

template <typename ScalarType, typename NonlinearSolverType>
Optimizer<ScalarType, NonlinearSolverType>::Optimizer(const optimizer_params_t& params,
                                                      const std::vector<Factor<Scalar>>& factors,
                                                      const Scalar epsilon,
                                                      const std::vector<Key>& keys,
                                                      const std::string& name, bool debug_stats,
                                                      bool check_derivatives)
    : factors_(factors),
      nonlinear_solver_(params, name, epsilon),
      epsilon_(epsilon),
      debug_stats_(debug_stats),
      keys_(keys.empty() ? ComputeKeysToOptimize(factors_, &Key::LexicalLessThan) : keys),
      index_(),
      linearizer_(factors_, keys_),
      linearize_func_(BuildLinearizeFunc(check_derivatives)) {
  stats_.iterations.reserve(params.iterations);
}

template <typename ScalarType, typename NonlinearSolverType>
Optimizer<ScalarType, NonlinearSolverType>::Optimizer(const optimizer_params_t& params,
                                                      std::vector<Factor<Scalar>>&& factors,
                                                      const Scalar epsilon, std::vector<Key>&& keys,
                                                      const std::string& name, bool debug_stats,
                                                      bool check_derivatives)
    : factors_(std::move(factors)),
      nonlinear_solver_(params, name, epsilon),
      epsilon_(epsilon),
      debug_stats_(debug_stats),
      keys_(keys.empty() ? ComputeKeysToOptimize(factors_, &Key::LexicalLessThan)
                         : std::move(keys)),
      index_(),
      linearizer_(factors_, keys_),
      linearize_func_(BuildLinearizeFunc(check_derivatives)) {
  stats_.iterations.reserve(params.iterations);
}

// ----------------------------------------------------------------------------
// Public methods
// ----------------------------------------------------------------------------

template <typename ScalarType, typename NonlinearSolverType>
bool Optimizer<ScalarType, NonlinearSolverType>::Optimize(Values<Scalar>* values,
                                                          int num_iterations) {
  if (num_iterations < 0) {
    num_iterations = nonlinear_solver_.Params().iterations;
  }

  Initialize(*values);

  // Clear state for this run
  nonlinear_solver_.Reset(*values);
  stats_.iterations.clear();

  return IterateToConvergence(values, num_iterations);
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
void Optimizer<ScalarType, NonlinearSolverType>::ComputeCovariancesAtBest(
    std::unordered_map<Key, MatrixX<Scalar>>* const covariances_by_key) {
  nonlinear_solver_.ComputeCovarianceAtBest(&covariance_);
  linearizer_.SplitCovariancesByKey(covariance_, covariances_by_key);
}

template <typename ScalarType, typename NonlinearSolverType>
const std::vector<Key>& Optimizer<ScalarType, NonlinearSolverType>::Keys() const {
  return keys_;
}

template <typename ScalarType, typename NonlinearSolverType>
const optimization_stats_t& Optimizer<ScalarType, NonlinearSolverType>::Stats() const {
  return stats_;
}

template <typename ScalarType, typename NonlinearSolverType>
void Optimizer<ScalarType, NonlinearSolverType>::UpdateParams(const optimizer_params_t& params) {
  nonlinear_solver_.UpdateParams(params);
}

// ----------------------------------------------------------------------------
// Protected methods
// ----------------------------------------------------------------------------

template <typename ScalarType, typename NonlinearSolverType>
bool Optimizer<ScalarType, NonlinearSolverType>::IterateToConvergence(Values<Scalar>* const values,
                                                                      const size_t num_iterations) {
  bool converged = false;

  // Iterate
  for (int i = 0; i < num_iterations; i++) {
    const bool early_exit = nonlinear_solver_.Iterate(linearize_func_, &stats_, debug_stats_);
    if (early_exit) {
      converged = true;
      break;
    }
  }

  // Save best results
  (*values) = nonlinear_solver_.GetBestValues();
  return converged;
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

}  // namespace sym
