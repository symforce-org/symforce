/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include "./levenberg_marquardt_solver.h"
#include "./linearizer.h"
#include "./optimization_stats.h"

namespace sym {

/**
 * Class for optimizing a nonlinear least-squares problem specified as a list of Factors.  For
 * efficient use, create once and call Optimize() multiple times with different initial guesses, as
 * long as the factors remain constant and the structure of the Values is identical.
 *
 * Not thread safe! Create one per thread.
 *
 * Example usage:
 *
 *   // Create a Values
 *   sym::Key key0{'R', 0};
 *   sym::Key key1{'R', 1};
 *   sym::Valuesd values;
 *   values.Set(key0, sym::Rot3d::Identity());
 *   values.Set(key1, sym::Rot3d::Identity());
 *
 *   // Create some factors
 *   std::vector<sym::Factord> factors;
 *   factors.push_back(sym::Factord::Jacobian(
 *       [epsilon](const sym::Rot3d& rot, Eigen::Vector3d* const res, Eigen::Matrix3d* const jac) {
 *         const sym::Rot3d prior = sym::Rot3d::Random();
 *         const Eigen::Matrix3d sqrt_info = Eigen::Vector3d::Ones().asDiagonal();
 *         sym::PriorFactorRot3<double>(rot, prior, sqrt_info, epsilon, res, jac);
 *       },
 *       {key0}));
 *   factors.push_back(sym::Factord::Jacobian(
 *       [epsilon](const sym::Rot3d& rot, Eigen::Vector3d* const res, Eigen::Matrix3d* const jac) {
 *         const sym::Rot3d prior = sym::Rot3d::Random();
 *         const Eigen::Matrix3d sqrt_info = Eigen::Vector3d::Ones().asDiagonal();
 *         sym::PriorFactorRot3<double>(rot, prior, sqrt_info, epsilon, res, jac);
 *       },
 *       {key1}));
 *   factors.push_back(sym::Factord::Jacobian(
 *       [epsilon](const sym::Rot3d& a, const sym::Rot3d& b, Eigen::Vector3d* const res,
 *                 Eigen::Matrix<double, 3, 6>* const jac) {
 *         const Eigen::Matrix3d sqrt_info = Eigen::Vector3d::Ones().asDiagonal();
 *         const sym::Rot3d a_T_b = sym::Rot3d::Random();
 *         sym::BetweenFactorRot3<double>(a, b, a_T_b, sqrt_info, epsilon, res, jac);
 *       },
 *       {key0, key1}));
 *
 *   // Set up the params
 *   sym::optimizer_params_t params = DefaultLmParams();
 *   params.iterations = 50;
 *   params.early_exit_min_reduction = 0.0001;
 *
 *   // Optimize
 *   sym::Optimizer<double> optimizer(params, factors, epsilon);
 *   optimizer.Optimize(&values);
 *
 * See symforce/test/symforce_optimizer_test.cc for more examples
 */
template <typename ScalarType, typename NonlinearSolverType = LevenbergMarquardtSolver<ScalarType>>
class Optimizer {
 public:
  using Scalar = ScalarType;
  using NonlinearSolver = NonlinearSolverType;

  /**
   * Constructor that copies in factors and keys
   */
  Optimizer(const optimizer_params_t& params, const std::vector<Factor<Scalar>>& factors,
            const Scalar epsilon = 1e-9, const std::string& name = "sym::Optimize",
            const std::vector<Key>& keys = {}, bool debug_stats = false,
            bool check_derivatives = false);

  /**
   * Constructor that copies in factors and keys, with arguments for the nonlinear solver
   */
  template <typename... NonlinearSolverArgs>
  Optimizer(const optimizer_params_t& params, const std::vector<Factor<Scalar>>& factors,
            const Scalar epsilon, const std::string& name, const std::vector<Key>& keys,
            bool debug_stats, bool check_derivatives, NonlinearSolverArgs&&... args);

  /**
   * Constructor with move constructors for factors and keys.
   */
  Optimizer(const optimizer_params_t& params, std::vector<Factor<Scalar>>&& factors,
            const Scalar epsilon = 1e-9, const std::string& name = "sym::Optimize",
            std::vector<Key>&& keys = {}, bool debug_stats = false, bool check_derivatives = false);

  /**
   * Constructor with move constructors for factors and keys, with arguments for the nonlinear
   * solver
   */
  template <typename... NonlinearSolverArgs>
  Optimizer(const optimizer_params_t& params, std::vector<Factor<Scalar>>&& factors,
            const Scalar epsilon, const std::string& name, std::vector<Key>&& keys,
            bool debug_stats, bool check_derivatives, NonlinearSolverArgs&&... args);

  // This cannot be moved or copied because the linearization keeps a pointer to the factors
  Optimizer(Optimizer&&) = delete;
  Optimizer& operator=(Optimizer&&) = delete;
  Optimizer(const Optimizer&) = delete;
  Optimizer& operator=(const Optimizer&) = delete;

  virtual ~Optimizer() = default;

  /**
   * Optimize the given values in-place
   *
   * Args:
   *     num_iterations: If < 0 (the default), uses the number of iterations specified by the params
   *                     at construction
   *     populate_best_linearization: If true, the linearization at the best values will be filled
   *                                  out in the stats
   *
   * Returns:
   *     The optimization stats
   */
  OptimizationStats<Scalar> Optimize(Values<Scalar>* values, int num_iterations = -1,
                                     bool populate_best_linearization = false);

  /**
   * Optimize the given values in-place
   *
   * This overload takes the stats as an argument, and stores into there.  This allows users to
   * avoid reallocating memory for any of the entries in the stats, for use cases where that's
   * important.  If passed, stats must not be nullptr.
   *
   * Args:
   *     num_iterations: If < 0 (the default), uses the number of iterations specified by the params
   *                     at construction
   *     populate_best_linearization: If true, the linearization at the best values will be filled
   *                                  out in the stats
   *     stats: An OptimizationStats to fill out with the result - if filling out dynamically
   *            allocated fields here, will not reallocate if memory is already allocated in the
   *            required shape (e.g. for repeated calls to Optimize)
   */
  virtual void Optimize(Values<Scalar>* values, int num_iterations,
                        bool populate_best_linearization, OptimizationStats<Scalar>* stats);

  /**
   * Optimize the given values in-place
   *
   * This overload takes the stats as an argument, and stores into there.  This allows users to
   * avoid reallocating memory for any of the entries in the stats, for use cases where that's
   * important.  If passed, stats must not be nullptr.
   *
   * Args:
   *     num_iterations: If < 0 (the default), uses the number of iterations specified by the params
   *                     at construction
   *     stats: An OptimizationStats to fill out with the result - if filling out dynamically
   *            allocated fields here, will not reallocate if memory is already allocated in the
   *            required shape (e.g. for repeated calls to Optimize)
   */
  void Optimize(Values<Scalar>* values, int num_iterations, OptimizationStats<Scalar>* stats);

  /**
   * Optimize the given values in-place
   *
   * This overload takes the stats as an argument, and stores into there.  This allows users to
   * avoid reallocating memory for any of the entries in the stats, for use cases where that's
   * important.  If passed, stats must not be nullptr.
   *
   * Args:
   *     stats: An OptimizationStats to fill out with the result - if filling out dynamically
   *            allocated fields here, will not reallocate if memory is already allocated in the
   *            required shape (e.g. for repeated calls to Optimize)
   */
  void Optimize(Values<Scalar>* values, OptimizationStats<Scalar>* stats);

  /**
   * Linearize the problem around the given values
   */
  Linearization<Scalar> Linearize(const Values<Scalar>& values);

  /**
   * Get covariances for each optimized key at the given linearization
   *
   * Will reuse entries in covariances_by_key, allocating new entries so that the result contains
   * exactly the set of keys optimized by this Optimizer.  `covariances_by_key` must not contain any
   * keys that are not optimized by this Optimizer.
   *
   * May not be called before either Optimize or Linearize has been called.
   */
  void ComputeAllCovariances(const Linearization<Scalar>& linearization,
                             std::unordered_map<Key, MatrixX<Scalar>>* covariances_by_key);

  /**
   * Get covariances for the given subset of keys at the given linearization.  This version is
   * potentially much more efficient than computing the covariances for all keys in the problem.
   *
   * Currently requires that `keys` corresponds to a set of keys at the start of the list of keys
   * for the full problem, and in the same order.  It uses the Schur complement trick, so will be
   * most efficient if the hessian is of the following form, with C block diagonal:
   *
   *     A = ( B    E )
   *         ( E^T  C )
   *
   * Will reuse entries in covariances_by_key, allocating new entries so that the result contains
   * exactly the set of keys requested.  `covariances_by_key` must not contain any keys that are not
   * in `keys`.
   */
  void ComputeCovariances(const Linearization<Scalar>& linearization, const std::vector<Key>& keys,
                          std::unordered_map<Key, MatrixX<Scalar>>* covariances_by_key);

  /**
   * Get the optimized keys
   */
  const std::vector<Key>& Keys() const;

  /**
   * Get the factors.
   */
  const std::vector<Factor<Scalar>>& Factors() const;

  /**
   * Get the Linearizer object
   */
  const sym::Linearizer<Scalar>& Linearizer() const;
  sym::Linearizer<Scalar>& Linearizer();

  /**
   * Update the optimizer params
   */
  void UpdateParams(const optimizer_params_t& params);

 protected:
  /**
   * Call nonlinear_solver_.Iterate on the given values (updating in place) until out of iterations
   * or converged
   */
  void IterateToConvergence(Values<Scalar>* values, size_t num_iterations,
                            bool populate_best_linearization, OptimizationStats<Scalar>* stats);

  /**
   * Build the linearize_func functor for the underlying nonlinear solver
   */
  typename NonlinearSolver::LinearizeFunc BuildLinearizeFunc(const bool check_derivatives);

  bool IsInitialized() const;

  /**
   * Do initialization that depends on having a values
   */
  void Initialize(const Values<Scalar>& values);

  const std::string& GetName();

  // Store a copy of the nonlinear factors. The Linearization object in the state keeps a
  // pointer to this memory.
  std::vector<Factor<Scalar>> factors_;

  // The name of this optimizer to be used for printing debug information.
  std::string name_;

  // Underlying nonlinear solver class.
  NonlinearSolver nonlinear_solver_;

  Scalar epsilon_;
  bool debug_stats_;

  std::vector<Key> keys_;
  index_t index_;

  sym::Linearizer<Scalar> linearizer_;

  // Covariance matrix and damped Hessian, only used by ComputeCovariances but cached here to save
  // reallocations. This may be the full problem covariance, or a subblock; it's always the full
  // problem Hessian
  struct ComputeCovariancesStorage {
    sym::MatrixX<Scalar> covariance;
    Eigen::SparseMatrix<Scalar> H_damped;
  };

  mutable ComputeCovariancesStorage compute_covariances_storage_;

  // Functor for interfacing with the optimizer
  typename NonlinearSolver::LinearizeFunc linearize_func_;
};

// Shorthand instantiations
using Optimizerd = Optimizer<double>;
using Optimizerf = Optimizer<float>;

/**
 * Simple wrapper to make it one function call.
 */
template <typename Scalar, typename NonlinearSolverType = LevenbergMarquardtSolver<Scalar>>
OptimizationStats<Scalar> Optimize(const optimizer_params_t& params,
                                   const std::vector<Factor<Scalar>>& factors,
                                   Values<Scalar>* values, const Scalar epsilon = 1e-9) {
  Optimizer<Scalar, NonlinearSolverType> optimizer(params, factors, epsilon);
  return optimizer.Optimize(values);
}

/**
 * Sensible default parameters for Optimizer
 */
optimizer_params_t DefaultOptimizerParams();

}  // namespace sym

#include "./optimizer.tcc"
