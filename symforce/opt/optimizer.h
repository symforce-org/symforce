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
  using Linearization = LinearizationWrapperLM<Scalar>;
  using LMOptimizer = levenberg_marquardt::Optimizer<Values<Scalar>, Linearization>;
  using LMOptimizerIterations = std::vector<levenberg_marquardt::levenberg_marquardt_iteration_t>;

  /**
   * Constructor that copies in factors and keys
   */
  Optimizer(const levenberg_marquardt::lm_params_t& params,
            const std::vector<Factor<Scalar>>& factors, const Scalar epsilon = 1e-9,
            const std::vector<Key>& keys = {}, const std::string& name = "sym::Optimize",
            bool debug_stats = false, bool check_derivatives = false);

  /**
   * Constructor with move constructors for factors and keys.
   */
  Optimizer(const levenberg_marquardt::lm_params_t& params, std::vector<Factor<Scalar>>&& factors,
            const Scalar epsilon = 1e-9, std::vector<Key>&& keys = {},
            const std::string& name = "sym::Optimize", bool debug_stats = false,
            bool check_derivatives = false);

  // This cannot be moved or copied because the linearization keeps a pointer to the factors
  Optimizer(Optimizer&&) = delete;
  Optimizer& operator=(Optimizer&&) = delete;
  Optimizer(const Optimizer&) = delete;
  Optimizer& operator=(const Optimizer&) = delete;

  /**
   * Optimize the given values in-place
   *
   * If num_iterations < 0 (the default), uses the number of iterations specified by the LM
   * Optimizer params at construction
   */
  bool Optimize(Values<Scalar>* values, int num_iterations = -1);

  /**
   * Continue optimizing, starting from the given values but not clearing the other optimizer state
   * (i.e. the iterations and lambda)
   *
   * If num_iterations < 0 (the default), uses the number of iterations specified by the LM
   * Optimizer params at construction
   */
  bool OptimizeContinue(Values<Scalar>* const values, int num_iterations = -1);

  /**
   * Linearize the problem around the given values
   */
  Linearization Linearize(const Values<Scalar>& values);

  /**
   * Get covariances for each optimized key at the current best values (requires that Optimize has
   * been called previously)
   */
  std::unordered_map<Key, Eigen::MatrixXd> ComputeCovariancesAtBest();

  /**
   * Get the optimized keys
   */
  const std::vector<Key>& Keys() const;

  /*
   * Get the underlying LM Optimizer state
   */
  const levenberg_marquardt::State<Values<Scalar>, LinearizationWrapperLM<Scalar>>&
  LMOptimizerState() const;

  /**
   * Get the LM Optimizer iterations
   */
  const LMOptimizerIterations& IterationStats() const;

  /**
   * Update the underlying LM optimizer params
   */
  void UpdateLMOptimizerParams(const levenberg_marquardt::lm_params_t& params);

 private:
  /**
   * Call lm_optimize_.Iterate on the given values (updating in place) until out of iterations or
   * converged
   */
  bool IterateToConvergence(Values<Scalar>* const values, const size_t num_iterations);

  /**
   * Build the linearize_func functor for the LM Optimizer
   */
  static typename LMOptimizer::LinearizeFunc BuildLinearizeFunc(
      sym::Optimizer<Scalar>* const optimizer, const index_t& index,
      const std::vector<sym::Factor<Scalar>>& factors, const std::vector<sym::Key>& keys,
      const Scalar epsilon, const bool check_derivatives);

  bool IsInitialized() const;

  /**
   * Do initialization that depends on having a values
   */
  void Initialize(const Values<Scalar>& values);

  /**
   * Helper to get an already initialized sym::Linearization object, linearized around the given
   * values.  Used to initialize multiple blocks in the LM Optimizer state without reanalyzing the
   * problem structure.
   */
  const sym::Linearization<Scalar>& GetInitializedLinearization(
      const std::vector<sym::Factor<Scalar>>& factors, const Values<Scalar>& values,
      const std::vector<Key>& key_order = {});

  // Store a copy of the nonlinear factors. The Linearization object in the state keeps a
  // pointer to this memory.
  std::vector<Factor<Scalar>> factors_;

  // Underlying optimizer class.
  LMOptimizer lm_optimizer_;

  // State block for the optimizer.
  levenberg_marquardt::State<Values<Scalar>, LinearizationWrapperLM<Scalar>> state_;

  // There are three state blocks inside the LM optimizer, each of which needs the indices for
  // linearization, which are identical.  So, this linearization is used to initialize each of them.
  // It will compute the indices once, but it will be relinearized around each block's initial
  // values when the block is created.  This linearization is also used for computing covariances
  // when requested.
  sym::Linearization<Scalar> linearization_;

  // Iteration stats
  LMOptimizerIterations iterations_;

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
