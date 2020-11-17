#pragma once

#include "./linearization_wrapper_lm.h"

namespace sym {

/**
 * Function that optimizes a nonlinear least-squares problem.
 *
 * We start with a nonlinear vector-valued error function that defines an error residual for
 * which we want to minimize the squared norm. The residual is dimension M, the state is N.
 *   residual = f(x)
 *
 * Define a least squares cost function as the squared norm of the residual:
 *   e(x) = 0.5 * ||f(x)||**2 = 0.5 * f(x).T * f(x)
 *
 * Take the first order taylor expansion for x around the linearization point x0:
 *   f(x) = f(x0) + f'(x0) * (x - x0) + ...
 *
 * Plug in to the cost function to get a quadratic:
 *   e(x) ~= 0.5 * (x - x0).T * f'(x0).T * f'(x0) * (x - x0) + f(x0).T * f'(x0) * (x - x0)
 *           + 0.5 * f(x0).T * f(x0)
 *
 * Take derivative with respect to x:
 *   e'(x) = f'(x0).T * f'(x0) * (x - x0) + f'(x0).T * f(x0)
 *
 * Set to zero to find the minimum value of the quadratic (paraboloid):
 *   0 = f'(x0).T * f'(x0) * (x - x0) + f'(x0).T * f(x0)
 *   (x - x0) = - inv(f'(x0).T * f'(x0)) * f'(x0).T * f(x0)
 *   x = x0 - inv(f'(x0).T * f'(x0)) * f'(x0).T * f(x0)
 *
 * Another way to write this is to create some helpful shorthand:
 *   f'(x0) --> jacobian or J (shape = MxN)
 *   f (x0) --> bias or b     (shape = Mx1)
 *   x - x0 --> dx            (shape = Nx1)
 *
 * Rederiving the Gauss-Newton solution:
 *   e(x) ~= 0.5 * dx.T * J.T * J * dx + b.T * J * dx + 0.5 * b.T * b
 *   e'(x) = J.T * J * dx + J.T * b
 *   x = x0 - inv(J.T * J) * J.T * b
 *
 * A couple more common names:
 *   f'(x0).T * f'(x0) = J.T * J --> hessian approximation or H (shape = NxN)
 *   f'(x0).T * f (x0) = J.T * b --> right hand side or rhs     (shape = Nx1)
 *
 * So the iteration loop for optimization is:
 *   J, b = linearize(f, x0)
 *   dx = -inv(J.T * J) * J.T * b
 *   x_new = x0 + dx
 */

/**
 * Class for second-order nonlinear optimization. For efficient use, create once and call Optimize()
 * multiple times with different initial guesses, as long as the factors remain constant and
 * the structure of the Values is identical.
 *
 * Not thread safe! Create one per thread.
 */
template <typename Scalar>
class Optimizer {
 public:
  using LMOptimizer =
      levenberg_marquardt::Optimizer<Values<Scalar>, LinearizationWrapperLM<Scalar>>;

  Optimizer(const levenberg_marquardt::lm_params_t& params,
            const std::vector<Factor<Scalar>>& factors, const Scalar epsilon = 1e-9,
            const std::vector<Key>& keys = {}, bool debug_stats = false)
      : factors_(factors),
        lm_optimizer_(params, "sym::Optimize"),
        epsilon_(epsilon),
        debug_stats_(debug_stats),
        keys_(keys.empty() ? ComputeKeysToOptimize(factors_, &Key::LexicalLessThan) : keys),
        linearize_func_(LinearizationWrapperLM<Scalar>::LinearizeFunc(factors_, keys_)) {
    iterations_.reserve(params.iterations);
  }

  /**
   * Version with move constructors for factors and keys.
   */
  Optimizer(const levenberg_marquardt::lm_params_t& params, std::vector<Factor<Scalar>>&& factors,
            const Scalar epsilon = 1e-9, std::vector<Key>&& keys = {}, bool debug_stats = false)
      : factors_(std::move(factors)),
        lm_optimizer_(params, "sym::Optimize"),
        epsilon_(epsilon),
        debug_stats_(debug_stats),
        keys_(keys.empty() ? ComputeKeysToOptimize(factors_, &Key::LexicalLessThan)
                           : std::move(keys)),
        linearize_func_(LinearizationWrapperLM<Scalar>::LinearizeFunc(factors_, keys_)) {
    iterations_.reserve(params.iterations);
  }

  void Optimize(Values<Scalar>* values) {
    // Initialization that depends on having a values
    if (index_.entries.size() == 0) {
      index_ = values->CreateIndex(keys_);
      update_func_ = LinearizationWrapperLM<Scalar>::UpdateFunc(index_, epsilon_);
    }

    // Clear state for this run
    lm_optimizer_.ResetState(*values, &state_);
    iterations_.clear();

    // Iterate
    for (int i = 0; i < lm_optimizer_.Params().iterations; i++) {
      const bool early_exit =
          lm_optimizer_.Iterate(linearize_func_, update_func_, &state_, &iterations_, debug_stats_);
      if (early_exit) {
        break;
      }
    }

    // Save best results
    (*values) = state_.Best().inputs;
  }

  const std::vector<levenberg_marquardt::levenberg_marquardt_iteration_t>& IterationStats() const {
    return iterations_;
  }

 private:
  // Store a copy of the nonlinear factors. The Linearization object in the state keeps a
  // pointer to this memory.
  std::vector<Factor<Scalar>> factors_;

  // Underlying optimizer class.
  LMOptimizer lm_optimizer_;

  // State block for the optimizer.
  // TODO(hayk): Right now there are three state blocks inside the LM optimizer, each of which
  // separately computes the indices for linearization, even though they are identical.
  levenberg_marquardt::State<Values<Scalar>, LinearizationWrapperLM<Scalar>> state_;

  // Iteration stats
  std::vector<levenberg_marquardt::levenberg_marquardt_iteration_t> iterations_;

  Scalar epsilon_;
  bool debug_stats_;

  std::vector<Key> keys_;
  index_t index_;

  // Functors for interfacing with the optimizer
  typename LMOptimizer::LinearizeFunc linearize_func_;
  typename LMOptimizer::UpdateFunc update_func_;
};

// Shorthand instantiations
using Optimizerd = Optimizer<double>;
using Optimizerf = Optimizer<float>;

/**
 * Simple wrapper to make it one function call.
 */
template <typename Scalar>
void Optimize(const levenberg_marquardt::lm_params_t& params,
              const std::vector<Factor<Scalar>>& factors, Values<Scalar>* values,
              const Scalar epsilon = 1e-9) {
  Optimizer<Scalar> optimizer(params, factors, epsilon);
  optimizer.Optimize(values);
}

}  // namespace sym
