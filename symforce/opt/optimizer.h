/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <sym/util/epsilon.h>

#include "./factor.h"
#include "./internal/linearizer_selector.h"
#include "./levenberg_marquardt_solver.h"
#include "./linearization.h"
#include "./optimization_stats.h"

namespace sym {

/**
 * Class for optimizing a nonlinear least-squares problem specified as a list of Factors.  For
 * efficient use, create once and call Optimize() multiple times with different initial
 * guesses, as long as the factors remain constant and the structure of the Values is identical.
 *
 * Not thread safe! Create one per thread.
 *
 * Example usage:
 *
 *     // Create a Values
 *     sym::Key key0{'R', 0};
 *     sym::Key key1{'R', 1};
 *     sym::Valuesd values;
 *     values.Set(key0, sym::Rot3d::Identity());
 *     values.Set(key1, sym::Rot3d::Identity());
 *
 *     // Create some factors
 *     std::vector<sym::Factord> factors;
 *     factors.push_back(sym::Factord::Jacobian(
 *         [epsilon](const sym::Rot3d& rot, Eigen::Vector3d* const res,
 *                   Eigen::Matrix3d* const jac) {
 *           const sym::Rot3d prior = sym::Rot3d::Random();
 *           const Eigen::Matrix3d sqrt_info = Eigen::Vector3d::Ones().asDiagonal();
 *           sym::PriorFactorRot3<double>(rot, prior, sqrt_info, epsilon, res, jac);
 *         },
 *         {key0}));
 *     factors.push_back(sym::Factord::Jacobian(
 *         [epsilon](const sym::Rot3d& rot, Eigen::Vector3d* const res,
 *                   Eigen::Matrix3d* const jac) {
 *           const sym::Rot3d prior = sym::Rot3d::Random();
 *           const Eigen::Matrix3d sqrt_info = Eigen::Vector3d::Ones().asDiagonal();
 *           sym::PriorFactorRot3<double>(rot, prior, sqrt_info, epsilon, res, jac);
 *         },
 *         {key1}));
 *     factors.push_back(sym::Factord::Jacobian(
 *         [epsilon](const sym::Rot3d& a, const sym::Rot3d& b, Eigen::Vector3d* const res,
 *                   Eigen::Matrix<double, 3, 6>* const jac) {
 *           const Eigen::Matrix3d sqrt_info = Eigen::Vector3d::Ones().asDiagonal();
 *           const sym::Rot3d a_T_b = sym::Rot3d::Random();
 *           sym::BetweenFactorRot3<double>(a, b, a_T_b, sqrt_info, epsilon, res, jac);
 *         },
 *         {key0, key1}));
 *
 *     // Set up the params
 *     sym::optimizer_params_t params = DefaultLmParams();
 *     params.iterations = 50;
 *     params.early_exit_min_reduction = 0.0001;
 *
 *     // Optimize
 *     sym::Optimizer<double> optimizer(params, factors, epsilon);
 *     optimizer.Optimize(values);
 *
 * See symforce/test/symforce_optimizer_test.cc for more examples
 */
template <typename ScalarType, typename NonlinearSolverType = LevenbergMarquardtSolver<ScalarType>>
class Optimizer {
 public:
  using Scalar = ScalarType;
  using NonlinearSolver = NonlinearSolverType;
  using FailureReason = typename NonlinearSolver::FailureReason;
  using MatrixType = typename NonlinearSolver::MatrixType;
  using Stats = OptimizationStats<MatrixType>;
  using LinearizerType = internal::LinearizerSelector_t<MatrixType>;

  /**
   * Base constructor
   *
   * @param params: The params to use for the optimizer and nonlinear solver
   * @param factors: The set of factors to include
   * @param name: The name of this optimizer to be used for printing debug information
   * @param keys: The set of keys to optimize.  If empty, will use all optimized keys touched by the
   *    factors
   * @param epsilon: Epsilon for numerical stability
   */
  Optimizer(const optimizer_params_t& params, std::vector<Factor<Scalar>> factors,
            const std::string& name = "sym::Optimize", std::vector<Key> keys = {},
            const Scalar epsilon = sym::kDefaultEpsilon<Scalar>);

  [[deprecated("Use the constructor that takes a params struct")]] Optimizer(
      const optimizer_params_t& params, std::vector<Factor<Scalar>> factors, const Scalar epsilon,
      const std::string& name, std::vector<Key> keys, bool debug_stats,
      bool check_derivatives = false, bool include_jacobians = false);

  /**
   * Constructor that copies in factors and keys, with arguments for the nonlinear solver
   *
   * See the base constructor for argument descriptions, additional arguments are forwarded to the
   * constructor for NonlinearSolverType
   */
  template <typename... NonlinearSolverArgs>
  Optimizer(const optimizer_params_t& params, std::vector<Factor<Scalar>> factors,
            const std::string& name, std::vector<Key> keys, Scalar epsilon,
            NonlinearSolverArgs&&... nonlinear_solver_args);

  template <typename... NonlinearSolverArgs>
  [[deprecated("Use the constructor that takes a params struct")]] Optimizer(
      const optimizer_params_t& params, std::vector<Factor<Scalar>> factors, Scalar epsilon,
      const std::string& name, std::vector<Key> keys, bool debug_stats, bool check_derivatives,
      bool include_jacobians, NonlinearSolverArgs&&... nonlinear_solver_args);

  // This cannot be moved or copied because the linearization keeps a pointer to the factors
  Optimizer(Optimizer&&) = delete;
  Optimizer& operator=(Optimizer&&) = delete;
  Optimizer(const Optimizer&) = delete;
  Optimizer& operator=(const Optimizer&) = delete;

  virtual ~Optimizer() = default;

  /**
   * Optimize the given values in-place
   *
   * @param num_iterations: If < 0 (the default), uses the number of iterations specified by the
   *    params at construction
   * @param populate_best_linearization: If true, the linearization at the best
   *    values will be filled out in the stats
   *
   * @returns The optimization stats
   */
  Stats Optimize(Values<Scalar>& values, int num_iterations = -1,
                 bool populate_best_linearization = false);

  /**
   * Optimize the given values in-place
   *
   * This overload takes the stats as an argument, and stores into there.  This allows users to
   * avoid reallocating memory for any of the entries in the stats, for use cases where that's
   * important.
   *
   * @param num_iterations: If < 0 (the default), uses the number of iterations specified by the
   *    params at construction
   * @param populate_best_linearization: If true, the linearization at the best values will be
   *    filled out in the stats
   * @param stats: An OptimizationStats to fill out with the result - if filling out dynamically
   *    allocated fields here, will not reallocate if memory is already allocated in the required
   *    shape (e.g. for repeated calls to Optimize())
   */
  virtual void Optimize(Values<Scalar>& values, int num_iterations,
                        bool populate_best_linearization, Stats& stats);

  /**
   * Optimize the given values in-place
   *
   * This overload takes the stats as an argument, and stores into there.  This allows users to
   * avoid reallocating memory for any of the entries in the stats, for use cases where that's
   * important.
   *
   * @param num_iterations: If < 0 (the default), uses the number of iterations specified by the
   *    params at construction
   * @param stats: An OptimizationStats to fill out with the result - if filling out dynamically
   *    allocated fields here, will not reallocate if memory is already allocated in the
   *    required shape (e.g. for repeated calls to Optimize())
   */
  void Optimize(Values<Scalar>& values, int num_iterations, Stats& stats);

  /**
   * Optimize the given values in-place
   *
   * This overload takes the stats as an argument, and stores into there.  This allows users to
   * avoid reallocating memory for any of the entries in the stats, for use cases where that's
   * important.
   *
   * @param stats: An OptimizationStats to fill out with the result - if filling out dynamically
   *    allocated fields here, will not reallocate if memory is already allocated in the
   *    required shape (e.g. for repeated calls to Optimize())
   */
  void Optimize(Values<Scalar>& values, Stats& stats);

  /**
   * Linearize the problem around the given values
   */
  Linearization<MatrixType> Linearize(const Values<Scalar>& values);

  /**
   * Get covariances for each optimized key at the given linearization
   *
   * Will reuse entries in covariances_by_key, allocating new entries so that the result contains
   * exactly the set of keys optimized by this Optimizer.  `covariances_by_key` must not contain any
   * keys that are not optimized by this Optimizer.
   *
   * May not be called before either Optimize() or Linearize() has been called.
   */
  void ComputeAllCovariances(const Linearization<MatrixType>& linearization,
                             std::unordered_map<Key, MatrixX<Scalar>>& covariances_by_key);

  /**
   * Get covariances for the given subset of keys at the given linearization
   *
   * This version is potentially much more efficient than computing the covariances for all keys in
   * the problem.
   *
   * Currently requires that `keys` corresponds to a set of keys at the start of the list of keys
   * for the full problem, and in the same order.  It uses the Schur complement trick, so will be
   * most efficient if the hessian is of the following form, with C block diagonal:
   *
   *     A = ( B    E )
   *         ( E^T  C )
   *
   * Will reuse entries in `covariances_by_key`, allocating new entries so that the result contains
   * exactly the set of keys requested.  `covariances_by_key` must not contain any keys that are not
   * in `keys`.
   */
  void ComputeCovariances(const Linearization<MatrixType>& linearization,
                          const std::vector<Key>& keys,
                          std::unordered_map<Key, MatrixX<Scalar>>& covariances_by_key);

  /**
   * Get the full problem covariance at the given linearization
   *
   * Unlike ComputeCovariance and ComputeAllCovariances, this includes the off-diagonal blocks, i.e.
   * the cross-covariances between different keys.
   *
   * The ordering of entries here is the same as the ordering of the keys in the linearization,
   * which can be accessed via ``optimizer.Linearizer().StateIndex()``.
   *
   * May not be called before either Optimize() or Linearize() has been called.
   *
   * @param covariance A matrix that will be filled out with the full problem covariance.
   */
  void ComputeFullCovariance(const Linearization<MatrixType>& linearization,
                             MatrixX<Scalar>& covariance);

  /**
   * Get the optimized keys
   */
  const std::vector<Key>& Keys() const;

  /**
   * Get the factors
   */
  const std::vector<Factor<Scalar>>& Factors() const;

  /**
   * Get the Linearizer object
   */
  const LinearizerType& Linearizer() const;
  LinearizerType& Linearizer();

  /**
   * Update the optimizer params
   */
  void UpdateParams(const optimizer_params_t& params);

  /**
   * Get the params used by the nonlinear solver
   */
  const optimizer_params_t& Params() const;

 protected:
  /**
   * Call nonlinear_solver_.Iterate() on the given values (updating in place) until out of
   * iterations or converged
   */
  void IterateToConvergence(Values<Scalar>& values, int num_iterations,
                            bool populate_best_linearization, Stats& stats);

  /**
   * Build the `linearize_func` functor for the underlying nonlinear solver
   */
  typename NonlinearSolver::LinearizeFunc BuildLinearizeFunc(bool check_derivatives);

  bool IsInitialized() const;

  /**
   * Do initialization that depends on having a values
   */
  void Initialize(const Values<Scalar>& values);

  const std::string& GetName();

  /// Store a copy of the nonlinear factors. The Linearization object in the state keeps a
  /// pointer to this memory.
  std::vector<Factor<Scalar>> factors_;

  /// The name of this optimizer to be used for printing debug information.
  std::string name_;

  /// Underlying nonlinear solver class.
  NonlinearSolver nonlinear_solver_;

  Scalar epsilon_;
  bool debug_stats_;
  bool include_jacobians_;

  std::vector<Key> keys_;
  index_t index_;

  LinearizerType linearizer_;

  /*
   * Covariance matrix and damped Hessian, only used by ComputeCovariances() but cached here to save
   * reallocations. This may be the full problem covariance, or a subblock; it's always the full
   * problem Hessian
   */
  struct ComputeCovariancesStorage {
    sym::MatrixX<Scalar> covariance;
    MatrixType H_damped;
  };

  mutable ComputeCovariancesStorage compute_covariances_storage_;

  /// Functor for interfacing with the optimizer
  typename NonlinearSolver::LinearizeFunc linearize_func_;

  bool verbose_;
};

// Shorthand instantiations
using Optimizerd = Optimizer<double>;
using Optimizerf = Optimizer<float>;

/**
 * Simple wrapper to make it one function call.
 */
template <typename Scalar, typename NonlinearSolverType = LevenbergMarquardtSolver<Scalar>>
typename Optimizer<Scalar, NonlinearSolverType>::Stats Optimize(
    const optimizer_params_t& params, const std::vector<Factor<Scalar>>& factors,
    Values<Scalar>& values, const Scalar epsilon = kDefaultEpsilon<Scalar>) {
  Optimizer<Scalar, NonlinearSolverType> optimizer(params, factors, "sym::Optimize", {}, epsilon);
  return optimizer.Optimize(values);
}

/**
 * Sensible default parameters for Optimizer
 */
optimizer_params_t DefaultOptimizerParams();

}  // namespace sym

#include "./optimizer.tcc"

// Explicit instantiation declaration
extern template class sym::Optimizer<double>;
extern template class sym::Optimizer<float>;
