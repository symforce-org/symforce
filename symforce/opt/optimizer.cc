#include "./optimizer.h"

namespace sym {

/**
 * Constructors
 */

template <typename Scalar>
Optimizer<Scalar>::Optimizer(const levenberg_marquardt::lm_params_t& params,
                             const std::vector<Factor<Scalar>>& factors, const Scalar epsilon,
                             const std::vector<Key>& keys, bool debug_stats)
    : factors_(factors),
      lm_optimizer_(params, "sym::Optimize"),
      epsilon_(epsilon),
      debug_stats_(debug_stats),
      keys_(keys.empty() ? ComputeKeysToOptimize(factors_, &Key::LexicalLessThan) : keys),
      index_(),
      linearize_func_(BuildLinearizeFunc(this, index_, factors_, keys_, epsilon)) {
  iterations_.reserve(params.iterations);
}

template <typename Scalar>
Optimizer<Scalar>::Optimizer(const levenberg_marquardt::lm_params_t& params,
                             std::vector<Factor<Scalar>>&& factors, const Scalar epsilon,
                             std::vector<Key>&& keys, bool debug_stats)
    : factors_(std::move(factors)),
      lm_optimizer_(params, "sym::Optimize"),
      epsilon_(epsilon),
      debug_stats_(debug_stats),
      keys_(keys.empty() ? ComputeKeysToOptimize(factors_, &Key::LexicalLessThan)
                         : std::move(keys)),
      index_(),
      linearize_func_(BuildLinearizeFunc(this, index_, factors_, keys_, epsilon)) {
  iterations_.reserve(params.iterations);
}

/**
 * Public methods
 */

template <typename Scalar>
bool Optimizer<Scalar>::Optimize(Values<Scalar>* values, int num_iterations) {
  if (num_iterations < 0) {
    num_iterations = lm_optimizer_.Params().iterations;
  }

  Initialize(*values);

  // Clear state for this run
  lm_optimizer_.ResetState(*values, &state_);
  iterations_.clear();

  bool converged = false;

  // Iterate
  for (int i = 0; i < num_iterations; i++) {
    const bool early_exit =
        lm_optimizer_.Iterate(linearize_func_, update_func_, &state_, &iterations_, debug_stats_);
    if (early_exit) {
      converged = true;
      break;
    }
  }

  // Save best results
  (*values) = state_.Best().inputs;
  return converged;
}

template <typename Scalar>
typename Optimizer<Scalar>::Linearization Optimizer<Scalar>::Linearize(
    const Values<Scalar>& values) {
  Initialize(values);

  LinearizationWrapperLM<Scalar> linearization;
  linearize_func_(values, &linearization);
  return linearization;
}

template <typename Scalar>
const std::vector<Key>& Optimizer<Scalar>::Keys() const {
  return keys_;
}

template <typename Scalar>
const levenberg_marquardt::State<Values<Scalar>, LinearizationWrapperLM<Scalar>>&
Optimizer<Scalar>::LMOptimizerState() const {
  return state_;
}

template <typename Scalar>
const typename Optimizer<Scalar>::LMOptimizerIterations& Optimizer<Scalar>::IterationStats() const {
  return iterations_;
}

/**
 * Private methods
 */

template <typename Scalar>
typename Optimizer<Scalar>::LMOptimizer::LinearizeFunc Optimizer<Scalar>::BuildLinearizeFunc(
    sym::Optimizer<Scalar>* const optimizer, const index_t& index,
    const std::vector<sym::Factor<Scalar>>& factors, const std::vector<sym::Key>& keys,
    const Scalar epsilon) {
  return LinearizationWrapperLM<Scalar>::LinearizeFunc(
      index,
      std::bind(&sym::Optimizer<Scalar>::GetInitializedLinearization, optimizer, std::cref(factors),
                std::placeholders::_1, std::cref(keys)),
      epsilon);
}

template <typename Scalar>
void Optimizer<Scalar>::Initialize(const Values<Scalar>& values) {
  if (index_.entries.size() == 0) {
    index_ = values.CreateIndex(keys_);
    update_func_ = LinearizationWrapperLM<Scalar>::UpdateFunc(index_, epsilon_);
  }
}

template <typename Scalar>
const sym::Linearization<Scalar>& Optimizer<Scalar>::GetInitializedLinearization(
    const std::vector<sym::Factor<Scalar>>& factors, const Values<Scalar>& values,
    const std::vector<Key>& key_order) {
  if (!linearization_.IsInitialized()) {
    linearization_ = sym::Linearization<Scalar>(factors, values, key_order);
  } else {
    // NOTE(aaron): Initializing a Linearization requires linearizing around some values, so to
    // avoid linearizing twice in some cases, this function is expected to return a Linearization
    // that not only has the correct sparsity but is also linearized around the given values
    linearization_.Relinearize(values);
  }
  return linearization_;
}

// Instantiate Optimizers
template class Optimizer<double>;

// TODO(aaron):  Currently the LM Optimizer does not support floats, so this can't be instantiated
// template class Optimizer<float>;

}  // namespace sym
